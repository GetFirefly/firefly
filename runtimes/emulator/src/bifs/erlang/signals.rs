use std::mem;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use firefly_alloc::heap::Heap;
use firefly_rt::function::{ErlangResult, ModuleFunctionArity};
use firefly_rt::gc::{garbage_collect, Gc, RootSet};
use firefly_rt::process::monitor::{Monitor, MonitorEntry, MonitorFlags, UnaliasMode};
use firefly_rt::process::signals::Signal;
use firefly_rt::process::{Process, ProcessFlags, ProcessLock, StatusFlags, SystemTask, ARG0_REG};
use firefly_rt::scheduler::Scheduler;
use firefly_rt::services::registry::{self, Registrant, WeakAddress};
use firefly_rt::term::*;

use crate::badarg;
use crate::emulator::{current_scheduler, Action};

#[export_name = "erlang:unlink/1"]
pub extern "C-unwind" fn unlink(process: &mut ProcessLock, id: OpaqueTerm) -> ErlangResult {
    match id.into() {
        Term::Pid(pid) => {
            assert!(pid.is_local());
            if let Some(target) = registry::get_by_pid(&pid) {
                let addr = target.addr();
                let id = process.uniq;
                let mut id_used = false;
                if let Some(entry) = process.links.get(&addr) {
                    // Send unlink, but only if not already unlinking
                    id_used = true;
                    if entry.set_unlinking(id.get()) {
                        let result = target.send_signal(Signal::unlink(process.addr(), id));
                        if result.is_err() {
                            // The receiver is exiting, so go ahead and remove the link
                            process.links.unlink(&addr);
                        }
                    }
                }
                if id_used {
                    process.uniq = id.checked_add(1).unwrap();
                }
            }
            ErlangResult::Ok(true.into())
        }
        Term::Port(_) => todo!("unlinking ports is not yet implemented"),
        _ => badarg!(process, id),
    }
}

#[export_name = "erlang:alias/0"]
pub extern "C-unwind" fn alias0(process: &mut ProcessLock) -> ErlangResult {
    alias1(process, OpaqueTerm::NIL)
}

#[export_name = "erlang:alias/1"]
pub extern "C-unwind" fn alias1(process: &mut ProcessLock, mut opts: OpaqueTerm) -> ErlangResult {
    let heap_available = process.heap.heap_available();
    if heap_available < mem::size_of::<Reference>() {
        process.gc_needed = mem::size_of::<Reference>();
        let mut roots = RootSet::default();
        roots += &mut opts as *mut _;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let flags = match opts.into() {
        Term::Nil => MonitorFlags::empty() | UnaliasMode::default(),
        Term::Cons(alias_opts) => {
            let mut mode = UnaliasMode::default();
            for result in alias_opts.iter_raw() {
                if let Ok(result) = result {
                    if result.is_atom() {
                        let atom = result.as_atom();
                        if atom == atoms::ExplicitUnalias {
                            mode = UnaliasMode::Explicit;
                            continue;
                        }
                        if atom == atoms::Reply {
                            mode = UnaliasMode::ReplyDemonitor;
                            continue;
                        }
                    }
                }
                badarg!(process, opts)
            }

            MonitorFlags::empty() | mode
        }
        _ => badarg!(process, opts),
    };

    let reference_id = current_scheduler().next_reference_id();
    let reference = Reference::new_pid(reference_id, process.pid());
    let monitor_ref = Gc::new_in(reference.clone(), process).unwrap();

    let monitor = MonitorEntry::new(Monitor::Alias {
        origin: process.id(),
        reference,
    });
    monitor.set_flags(flags);
    process.monitored.insert(monitor.clone());

    ErlangResult::Ok(monitor_ref.into())
}

#[export_name = "erlang:unalias/1"]
pub extern "C-unwind" fn unalias(process: &mut ProcessLock, alias: OpaqueTerm) -> ErlangResult {
    if let Term::Reference(reference) = alias.into() {
        if let Some(pid) = reference.pid() {
            if process.pid() != pid {
                return ErlangResult::Ok(false.into());
            }

            let mut cursor = process.monitored.find_mut(&reference.id());
            if cursor.is_null() {
                return ErlangResult::Ok(false.into());
            }
            let monitor_entry = cursor.get().unwrap();
            if !monitor_entry.flags().contains(MonitorFlags::ALIAS) {
                return ErlangResult::Ok(false.into());
            }
            assert_eq!(monitor_entry.origin(), Some(WeakAddress::Process(pid)));
            monitor_entry.remove_flags(MonitorFlags::ALIAS_MASK);
            if let Monitor::Alias { .. } = monitor_entry.monitor {
                cursor.remove();
            }
            return ErlangResult::Ok(true.into());
        }
    }

    badarg!(process, alias)
}

static HANDLE_SIGNALS_TRAP_EXPORT: ModuleFunctionArity = ModuleFunctionArity {
    module: atoms::ErtsInternal,
    function: atoms::HandleSignals,
    arity: 2,
};

#[export_name = "erlang:is_process_alive/1"]
pub extern "C-unwind" fn is_process_alive(
    process: &mut ProcessLock,
    pid_term: OpaqueTerm,
) -> ErlangResult {
    static IS_PROCESS_ALIVE_TRAP_EXPORT: ModuleFunctionArity = ModuleFunctionArity {
        module: atoms::ErtsInternal,
        function: atoms::IsProcessAlive,
        arity: 1,
    };

    match pid_term.into() {
        Term::Pid(pid) if pid.is_local() => {
            if process.id() == pid.id() {
                return ErlangResult::Ok(true.into());
            }

            let result;
            match registry::get_by_pid(&pid) {
                None => result = false,
                Some(other) => {
                    let status = other.status(Ordering::Acquire);
                    if status.intersects(
                        StatusFlags::EXITING
                            | StatusFlags::HAS_PENDING_SIGNALS
                            | StatusFlags::HAS_IN_TRANSIT_SIGNALS,
                    ) {
                        // There are two conditions in which we need to wait:
                        //
                        // 1. If in exiting state, send `IsAlive` signal and wait for it to finish
                        // exiting
                        // 2. If the process has signals enqueued, we need to send it an `IsAlive`
                        // request in order to ensure that the signal order is preserved (we may
                        // have earlier sent it an exit signal that has not been processed yet).
                        process.stack.store(ARG0_REG, pid_term);
                        return ErlangResult::Trap(&IS_PROCESS_ALIVE_TRAP_EXPORT);
                    }

                    result = true;
                }
            }

            let status = process.status(Ordering::Acquire);
            if status
                .intersects(StatusFlags::HAS_PENDING_SIGNALS | StatusFlags::HAS_IN_TRANSIT_SIGNALS)
            {
                // Ensure that the signal order of signals from inspected process to us is preserved
                process.stack.store(ARG0_REG, pid_term);
                process.stack.store(ARG0_REG + 1, result.into());
                return ErlangResult::Trap(&HANDLE_SIGNALS_TRAP_EXPORT);
            }
            ErlangResult::Ok(result.into())
        }
        _ => badarg!(process, pid_term),
    }
}

#[export_name = "erts_internal:is_process_alive/2"]
pub extern "C-unwind" fn is_process_alive2(
    process: &mut ProcessLock,
    pid_term: OpaqueTerm,
    ref_term: OpaqueTerm,
) -> ErlangResult {
    let Term::Pid(pid) = pid_term.into() else { badarg!(process, pid_term); };
    let Term::Reference(req_ref) = ref_term.into() else { badarg!(process, ref_term); };

    let failed;
    match registry::get_by_pid(&pid) {
        None => failed = true,
        Some(other) => {
            let sig = Signal::is_alive(process.pid(), req_ref.deref().clone());
            failed = other.send_signal(sig).is_err();
        }
    }

    if failed {
        let status = process.status(Ordering::Acquire);
        if status.intersects(StatusFlags::HAS_PENDING_SIGNALS | StatusFlags::HAS_IN_TRANSIT_SIGNALS)
        {
            // Ensure that the signal order of signals from inspected process to us is preserved
            process.stack.store(ARG0_REG, pid_term);
            process.stack.store(ARG0_REG + 1, atoms::Ok.into());
            return ErlangResult::Trap(&HANDLE_SIGNALS_TRAP_EXPORT);
        }
    }

    ErlangResult::Ok(atoms::Ok.into())
}

#[export_name = "erts_internal:handle_signals/2"]
pub extern "C-unwind" fn handle_signals2(
    process: &mut ProcessLock,
    id_term: OpaqueTerm,
    result: OpaqueTerm,
) -> ErlangResult {
    use firefly_rt::process::signals::{FlushType, SignalQueueFlags};

    let mut sigq_flags = process.signals().flags();
    if sigq_flags.contains(SignalQueueFlags::FLUSHED) {
        assert!(sigq_flags.contains(SignalQueueFlags::FLUSHING));
        process
            .signals()
            .remove_flags(SignalQueueFlags::FLUSHED | SignalQueueFlags::FLUSHING);
        process.remove_flags(ProcessFlags::DISABLE_GC);
        return ErlangResult::Ok(result);
    }

    if !sigq_flags.contains(SignalQueueFlags::FLUSHING) {
        let id: Term = id_term.into();
        let flush_type = match id.try_into() {
            Ok(addr) => FlushType::Id(addr),
            Err(_) => FlushType::InTransit,
        };
        process.flush_signals(flush_type);
        sigq_flags = process.signals().flags();
        assert!(sigq_flags.contains(SignalQueueFlags::FLUSHING));
        if sigq_flags.contains(SignalQueueFlags::FLUSHED) {
            process
                .signals()
                .remove_flags(SignalQueueFlags::FLUSHED | SignalQueueFlags::FLUSHING);
            process.remove_flags(ProcessFlags::DISABLE_GC);
            return ErlangResult::Ok(result);
        }
    } else {
        assert!(sigq_flags.contains(SignalQueueFlags::FLUSHING));
    }

    let reds_left = process.reductions_left();
    let mut status = process.status(Ordering::Relaxed);
    match current_scheduler().handle_signals(process, &mut status, reds_left, true) {
        Action::Continue => {
            if status.contains(StatusFlags::EXITING) {
                process.stack.nocatch();
                return ErlangResult::Exit;
            }

            sigq_flags = process.signals().flags();
            if sigq_flags.contains(SignalQueueFlags::FLUSHED) {
                assert!(sigq_flags.contains(SignalQueueFlags::FLUSHING));
                process
                    .signals()
                    .remove_flags(SignalQueueFlags::FLUSHED | SignalQueueFlags::FLUSHING);
                process.remove_flags(ProcessFlags::DISABLE_GC);
                return ErlangResult::Ok(result);
            }

            // If we reach here, then we have processed all signals, but we're not yet flushed,
            // so we just yield back to the scheduler and try again
            ErlangResult::Trap(&HANDLE_SIGNALS_TRAP_EXPORT)
        }
        Action::Yield => {
            /*
             * More signals to handle, but out of reductions. Yield
             * and come back here and continue...
             */
            ErlangResult::Trap(&HANDLE_SIGNALS_TRAP_EXPORT)
        }
        Action::Error(e) => panic!("unexpected error occurred: {:?}", e),
        Action::Suspend | Action::Killed => unreachable!(),
    }
}

#[export_name = "erts_internal:request_system_task/3"]
pub extern "C-unwind" fn garbage_collect1(
    process: &mut ProcessLock,
    pid_term: OpaqueTerm,
    priority_term: OpaqueTerm,
    request_term: OpaqueTerm,
) -> ErlangResult {
    use firefly_rt::process::{Priority, SystemTaskType};

    let Term::Pid(pid) = pid_term.into() else { badarg!(process, pid_term); };

    let status = process.status(Ordering::Relaxed);
    let priority = if priority_term == atoms::Inherit {
        status.priority()
    } else {
        let prio: Term = priority_term.into();
        match Priority::try_from(prio) {
            Ok(prio) => prio,
            Err(_) => badarg!(process, priority_term),
        }
    };

    let target = registry::get_by_pid(&pid);

    // The request must be a tuple of 2 or 3 elements
    match request_term.tuple_size() {
        Ok(arity) => {
            if arity < 2 || arity > 4 {
                badarg!(process, request_term)
            }
        }
        // Not a tuple
        _ => badarg!(process, request_term),
    }

    let Term::Tuple(request) = request_term.into() else { unreachable!() };

    let requestor = process.pid();
    let request_type = request[0];
    let request_id: Term = request[1].into();
    let signal = pid == requestor;

    let mut system_task: Box<SystemTask>;
    match request_type.into() {
        Term::Atom(a) if a == atoms::GarbageCollect => {
            if request.len() < 3 {
                badarg!(process, request_term);
            }
            let ty_arg = request[2];
            let ty = match ty_arg.into() {
                Term::Atom(arg) if arg == atoms::Major => SystemTaskType::GcMajor,
                Term::Atom(arg) if arg == atoms::Minor => SystemTaskType::GcMinor,
                _ => badarg!(process, request_term),
            };
            system_task = SystemTask::new(ty, request_id.layout()).unwrap();
            system_task.requestor = requestor.into();
            system_task.priority = priority;
            system_task.reply_tag = request_type;
            let request_id = unsafe { request_id.unsafe_clone_to_heap(system_task.fragment()) };
            system_task.request_id = request_id.into();
            system_task.args[0] = ty_arg;
            if target.is_none() {
                notify_sys_task_executed(process, system_task, false.into());
                return ErlangResult::Ok(atoms::Ok.into());
            }
        }
        // We implement this so that tests for system tasks borrowed from ERTS can be run
        Term::Atom(a) if a == "system_task_test" => {
            if request.len() > 4 {
                badarg!(process, request_term);
            }
            let mut layout = LayoutBuilder::new();
            layout += request_id.layout();
            let mut args = [Term::None, Term::None];
            for arg in &request[2..] {
                args[0] = (*arg).into();
                layout += args[0].layout();
            }
            let layout = layout.finish();
            system_task = SystemTask::new(SystemTaskType::Test, layout).unwrap();
            system_task.requestor = requestor.into();
            system_task.priority = priority;
            system_task.reply_tag = request_type;
            let request_id = unsafe { request_id.unsafe_clone_to_heap(system_task.fragment()) };
            system_task.request_id = request_id.into();
            for (i, arg) in args.iter().enumerate() {
                let arg = unsafe { arg.unsafe_clone_to_heap(system_task.fragment()) };
                system_task.args[i] = arg.into();
            }
            if target.is_none() {
                notify_sys_task_executed(process, system_task, false.into());
                return ErlangResult::Ok(atoms::Ok.into());
            }
        }
        _ => badarg!(process, request_type),
    }

    let target = target.unwrap();

    if signal {
        let status = target.status(Ordering::Acquire);
        if status.contains(StatusFlags::EXITING) {
            notify_sys_task_executed(process, system_task, false.into());
            return ErlangResult::Ok(atoms::Ok.into());
        }
        if status.intersects(StatusFlags::HAS_PENDING_SIGNALS | StatusFlags::HAS_IN_TRANSIT_SIGNALS)
        {
            // Send rpc request signal without reply, and reply from the system task...
            let st = Box::into_raw(system_task);
            if target
                .send_signal(Signal::rpc_noreply(
                    process.pid(),
                    schedule_sig_sys_task,
                    st.cast(),
                    priority,
                ))
                .is_err()
            {
                notify_sys_task_executed(process, unsafe { Box::from_raw(st) }, false.into());
            }
            // Signal sent
            return ErlangResult::Ok(atoms::Ok.into());
        }
    }

    if let Err(system_task) = schedule_sys_task(process, target, system_task) {
        notify_sys_task_executed(process, system_task, false.into());
    }

    return ErlangResult::Ok(atoms::Ok.into());
}

fn schedule_sig_sys_task(process: &mut ProcessLock, st: *mut ()) -> TermFragment {
    let system_task: Box<SystemTask> = unsafe { Box::from_raw(st.cast()) };

    if let Err(system_task) = schedule_sys_task(process, process.strong(), system_task) {
        notify_sys_task_executed(process, system_task, false.into());
    }
    TermFragment {
        term: OpaqueTerm::NONE,
        fragment: None,
    }
}

fn schedule_sys_task(
    process: &mut ProcessLock,
    target: Arc<Process>,
    system_task: Box<SystemTask>,
) -> Result<(), Box<SystemTask>> {
    // If we're sending to ourselves, we do things a bit different
    if core::ptr::eq(process.as_ref(), target.as_ref()) {
        let status = process.status(Ordering::Acquire);
        // Not sure this is even possible, but we account for it anyway
        if status.intersects(StatusFlags::EXITING | StatusFlags::FREE) {
            return Err(system_task);
        }
        // We can push the task directly into the system task queue
        process.system_tasks[system_task.priority as usize].push_back(system_task);
        // We now have system tasks to execute, make sure our status reflects that
        process.set_status_flags(
            StatusFlags::ACTIVE_SYS | StatusFlags::SYS_TASKS,
            Ordering::Release,
        );
    } else {
        let mut status = target.status(Ordering::Acquire);
        loop {
            if status.contains(StatusFlags::EXITING | StatusFlags::FREE) {
                return Err(system_task);
            }
            match target.status.compare_exchange(
                status,
                status | StatusFlags::ACTIVE_SYS | StatusFlags::SYS_TASKS,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(prev) => {
                    // The process is currently suspended, we need to reschedule it
                    let tgt = target.clone();
                    let mut guard = target.lock();
                    guard.system_tasks[system_task.priority as usize].push_back(system_task);
                    // If currently suspended, wake up the process to handle the task
                    if prev.contains(StatusFlags::SUSPENDED) {
                        guard.injector.push(tgt);
                    }
                    break;
                }
                Err(current) => {
                    status = current;
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn notify_sys_task_executed(
    process: &mut ProcessLock,
    task: Box<SystemTask>,
    result: OpaqueTerm,
) {
    if let Some(Registrant::Process(requestor)) = task.requestor.try_resolve() {
        send_sys_task_executed_reply(
            process.pid(),
            requestor,
            task.reply_tag.into(),
            task.request_id.into(),
            result.into(),
        );
    }
}

fn send_sys_task_executed_reply(
    from: Pid,
    to: Arc<Process>,
    tag: Term,
    request_id: Term,
    result: Term,
) {
    let mut layout = LayoutBuilder::new();
    layout += tag.layout();
    layout += request_id.layout();
    layout += result.layout();
    layout.build_tuple(3);
    let fragment_ptr = layout.into_fragment().unwrap();
    let fragment = unsafe { fragment_ptr.as_ref() };

    let tag = unsafe { tag.unsafe_clone_to_heap(fragment) };
    let request_id = unsafe { tag.unsafe_clone_to_heap(fragment) };
    let result = unsafe { tag.unsafe_clone_to_heap(fragment) };
    let tuple =
        Tuple::from_slice(&[tag.into(), request_id.into(), result.into()], fragment).unwrap();

    to.send_fragment(
        from.into(),
        TermFragment {
            term: tuple.into(),
            fragment: Some(fragment_ptr),
        },
    )
    .ok();
}
