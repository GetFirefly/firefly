use std::alloc::{AllocError, Layout};
use std::intrinsics::{likely, unlikely};
use std::mem;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use firefly_alloc::fragment::HeapFragment;
use firefly_alloc::heap::Heap;
use firefly_bytecode::{self as bc, ops, Function, Opcode, Register};
use firefly_number::{Int, Number};
use firefly_rt::backtrace::{Trace, TraceFrame};
use firefly_rt::cmp::ExactEq;
use firefly_rt::error::{ErrorCode, ExceptionFlags, ExceptionInfo};
use firefly_rt::function::{self, DynamicCallee, ErlangResult, ModuleFunctionArity};
use firefly_rt::gc::{self, Gc};
use firefly_rt::process::link::{Link, LinkEntry, LinkTreeEntry};
use firefly_rt::process::monitor::{Monitor, MonitorEntry, MonitorFlags, MonitorTreeEntry};
use firefly_rt::process::signals::{
    self, Message, Signal, SignalEntry, SignalQueueFlags, SignalQueueLock,
};
use firefly_rt::process::{
    ContinueExitPhase, Process, ProcessFlags, ProcessLock, ProcessTimer, SpawnOpts, StatusFlags,
    ARG0_REG, CP_REG, RETURN_REG,
};
use firefly_rt::scheduler::{Scheduler, SchedulerId};
use firefly_rt::services::error_logger;
use firefly_rt::services::registry::{self, Registrant, WeakAddress};
use firefly_rt::services::timers::{Timer, TimerError, TimerService};
use firefly_rt::term::{
    atoms, BigInt, BinaryData, BitSlice, Closure, ClosureFlags, Cons, Map, MapError, MatchContext,
    OpaqueTerm, Pid, Reference, Term, Tuple, Value,
};
use firefly_rt::term::{LayoutBuilder, TermFragment, TermType};
use firefly_system::time::{Duration, Timeout};

use intrusive_collections::UnsafeRef;
use log::{log_enabled, trace};
use smallvec::{smallvec, SmallVec};

use crate::queue::TaskQueue;

use super::*;

type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

impl Scheduler for Emulator {
    fn id(&self) -> SchedulerId {
        self.id
    }

    fn thread_id(&self) -> std::thread::ThreadId {
        self.thread_id
    }

    fn next_reference_id(&self) -> ReferenceId {
        let reference_id = unsafe { &mut *self.reference_id.get() };
        let next = *reference_id;
        *reference_id += 1;
        unsafe { ReferenceId::new(self.id(), next) }
    }

    fn next_unique_integer(&self, positive: bool) -> Int {
        let unique_id = unsafe { &mut *self.unique_id.get() };
        let next = *unique_id;
        *unique_id += 1;
        crate::unique::make_unique_integer(self.id.as_u16() as u64, next, positive)
    }

    fn next_monotonic_integer(&self, positive: bool) -> Int {
        crate::unique::get_unique_monotonic_integer(positive)
    }

    fn start_timer(&self, timer: Timer) -> Result<(), TimerError> {
        // We only allow timer management from processes running on the same scheduler
        assert_eq!(self.thread_id, std::thread::current().id());
        self.timers.borrow_mut().start_timer(timer)
    }

    fn cancel_timer(&self, timer_ref: ReferenceId) -> Result<(), ()> {
        // We only allow timer management from processes running on the same scheduler
        assert_eq!(self.thread_id, std::thread::current().id());
        self.timers.borrow_mut().cancel_timer(timer_ref)
    }

    /// Spawn a new process with the given module/function/arguments
    fn spawn(
        &self,
        parent: &mut ProcessLock,
        mfa: ModuleFunctionArity,
        args: &[OpaqueTerm],
        opts: SpawnOpts,
    ) -> (Arc<Process>, Option<Gc<Reference>>) {
        use firefly_rt::process::monitor::LocalMonitorInfo;

        let spawn_async = opts.spawn_async;
        let monitor = opts.monitor;
        let link = opts.link;
        let spawn_ref_id = self.next_reference_id();
        let spawn_ref;
        if opts.monitor.map(|mo| mo.alias.is_some()).unwrap_or(false) {
            spawn_ref =
                Some(Gc::new_in(Reference::new_pid(spawn_ref_id, parent.pid()), parent).unwrap());
        } else if opts.monitor.is_some() {
            spawn_ref = Some(Gc::new_in(Reference::new(spawn_ref_id), parent).unwrap());
        } else {
            spawn_ref = None;
        }

        if log_enabled!(target: "scheduler", log::Level::Trace) {
            let argv = args
                .iter()
                .map(|a| format!("{}", a))
                .collect::<SmallVec<[String; 4]>>();
            let argv_formatted = argv.join(",");
            trace!(target: "scheduler", "spawning process with mfa {} and args [{}], link={}, monitor={}", &mfa, argv_formatted, monitor.is_some(), link);
        }

        let proc = Process::new(
            self.id(),
            Some(parent.pid()),
            Some(
                parent
                    .group_leader()
                    .cloned()
                    .unwrap_or_else(|| parent.pid()),
            ),
            mfa,
            args,
            self.injector.clone(),
            opts,
        );

        {
            let mut spawned = proc.lock();
            if link {
                let link = LinkEntry::new(Link::LocalProcess {
                    origin: parent.id(),
                    target: spawned.id(),
                });
                assert!(parent.links.link(link.clone()).is_ok());
                assert!(spawned.links.linked_by(link).is_ok());
            }

            if let Some(monitor_opts) = monitor {
                let monitor = MonitorEntry::new(Monitor::LocalProcess {
                    origin: parent.id(),
                    target: spawned.id(),
                    info: LocalMonitorInfo {
                        reference: spawn_ref_id,
                        name_or_tag: TermFragment::new(monitor_opts.tag.into()).unwrap(),
                    },
                });
                monitor.set_flags(monitor_opts.flags);
                parent.monitored.insert(monitor.clone());
                spawned.monitored_by.push_back(monitor);
            }
            assert!(!spawn_async, "asynchronous spawns are not implemented yet");
        }

        registry::register_process(proc.clone());

        self.runq.push(proc.clone());

        (proc, spawn_ref)
    }

    fn reschedule(&self, process: Arc<Process>) {
        self.runq.push(process);
    }
}

const MAX_REDUCTIONS: usize = Process::MAX_REDUCTIONS;
const ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR: usize = 4;

impl Emulator {
    /// Run the scheduler core loop indefinitely or until an error occurs
    pub(super) fn run(&self) -> Result<(), EmulatorError> {
        loop {
            if !self.run_once()? {
                // There are no processes available, sleep for a few seconds
                // and try scheduling again. This avoids busy looping with no
                // work.
                if let Some(ms) = self.timers.borrow().skippable() {
                    trace!(target: "scheduler", "scheduler has no processes available to schedule, parking until next timer expires");
                    std::thread::park_timeout(Duration::from_millis(ms as u64));
                }
            }
        }
    }

    /// Run a single iteration of the scheduler core loop
    #[inline]
    fn run_once(&self) -> Result<bool, EmulatorError> {
        // Get the next scheduled process
        //
        // If there is no process, there is nothing more to do, so break out of the run loop
        'next: loop {
            match self.runq.pop() {
                Some(process) => {
                    trace!(target: "scheduler", "scheduling process {}", process.pid());
                    // Within this loop we attempt to schedule `process`.
                    //
                    // If `process` should not be scheduled, we break out to `next` to try again
                    // with the next process.
                    //
                    // When `process` is scheduled out, we break out of the `schedule` loop which
                    // drops us into the code which handles descheduling, after which we return to
                    // the caller as we've completed a single scheduling cycle
                    'schedule: loop {
                        // Determine if this process should be scheduled and update its state if so
                        let mut status = process.status(Ordering::Acquire);
                        trace!(target: "scheduler", "process has status {:?}", status);
                        loop {
                            let mut new_status = status;
                            // Active? (has work to do)
                            let is_active =
                                status.intersects(StatusFlags::ACTIVE | StatusFlags::ACTIVE_SYS);
                            // Suspended? (suspend does not effect active-sys)
                            let is_suspended = status
                                & (StatusFlags::SUSPENDED | StatusFlags::ACTIVE_SYS)
                                == StatusFlags::SUSPENDED;
                            // Already running?
                            let is_running =
                                status.intersects(StatusFlags::RUNNING | StatusFlags::RUNNING_SYS);
                            let run = is_active && !is_suspended && !is_running;

                            // Mark this process as running with the correct flag
                            if run {
                                trace!(target: "scheduler", "process will be run");
                                if status.contains(StatusFlags::ACTIVE_SYS) {
                                    new_status |= StatusFlags::RUNNING_SYS;
                                } else {
                                    new_status |= StatusFlags::RUNNING;
                                }
                            }
                            match process.cmpxchg_status_flags(status, new_status) {
                                Ok(current) => {
                                    status = current;
                                    if !run {
                                        trace!(target: "scheduler", "process will not be run, placing back in run queue");
                                        self.runq.push(process);
                                        break 'next;
                                    }
                                    break;
                                }
                                Err(current) => {
                                    status = current;
                                    continue;
                                }
                            }
                        }

                        // Acquire the main process lock
                        let mut process = process.lock();
                        // Make sure we're recorded as the owning scheduler for as long
                        // as we're executing this process. Processes may migrate schedulers
                        // any time they are scheduled but not executing, so we only set this
                        // value twice: once when a process is spawned, and each time a process
                        // gets executed.
                        process.set_scheduler(self.id);

                        if status.contains(StatusFlags::RUNNING_SYS) {
                            if status.intersects(
                                StatusFlags::HAS_IN_TRANSIT_SIGNALS
                                    | StatusFlags::HAS_PENDING_SIGNALS,
                            ) {
                                trace!(target: "scheduler", "process has pending signals");
                                // If we have dirty work scheduled we allow usage
                                // of all reductions since we need to handle all signals
                                // before doing dirty work.
                                //
                                // If a BIF is flushing signals; we also allow usage of
                                // all reductions since the BIF cannot continue execution
                                // until the flush completes.
                                let mut signal_reductions =
                                    MAX_REDUCTIONS.saturating_sub(process.reductions);
                                if status.contains(StatusFlags::ACTIVE)
                                    && !process
                                        .signals()
                                        .flags()
                                        .contains(SignalQueueFlags::FLUSHING)
                                {
                                    // We are active, i.e. have erlang work to do, and have no dirty
                                    // work and are not
                                    // flushing. Limit the amount of signal handling work we do.
                                    signal_reductions = MAX_REDUCTIONS / 40;
                                }
                                match self.handle_signals(
                                    &mut process,
                                    &mut status,
                                    signal_reductions,
                                    true,
                                ) {
                                    Action::Continue => (),
                                    Action::Yield => {
                                        trace!(target: "scheduler", "process is yielding");
                                        break;
                                    }
                                    Action::Suspend => {
                                        trace!(target: "scheduler", "process is suspending");
                                        process.set_status_flags(
                                            StatusFlags::SUSPENDED,
                                            Ordering::Release,
                                        );
                                        break;
                                    }
                                    Action::Error(e) => return Err(e),
                                    _ => unreachable!(),
                                }
                            }
                            if status & (StatusFlags::SYS_TASKS | StatusFlags::EXITING)
                                == StatusFlags::SYS_TASKS
                            {
                                // GC is normally never delayed when a process is scheduled out,
                                // but if it happens, we are not allowed to execute system tasks.
                                //
                                // Execution of system tasks is also not allowed if a BIF is
                                // flushing signals, since there are
                                // system tasks that might need to fetch from the outer
                                // signal queue.
                                if !process.flags.contains(ProcessFlags::DELAY_GC)
                                    && !process
                                        .signals()
                                        .flags()
                                        .contains(SignalQueueFlags::FLUSHING)
                                {
                                    let cost = self.execute_sys_tasks(&mut process, &mut status);
                                    process.reductions += cost;
                                }
                            }
                            if process.reductions > MAX_REDUCTIONS {
                                // Schedule out
                                trace!(target: "scheduler", "process has consumed its reduction budget, yielding..");
                                break 'schedule;
                            }
                            assert!(status.contains(StatusFlags::RUNNING_SYS));
                            assert!(!status.contains(StatusFlags::RUNNING));

                            loop {
                                let mut new_status;

                                if (status & (StatusFlags::SUSPENDED | StatusFlags::ACTIVE)
                                    != StatusFlags::ACTIVE)
                                    && !status.contains(StatusFlags::EXITING)
                                {
                                    // Handling signals and system tasks may have created
                                    // data on the process heap that should be GC'd
                                    if process.is_gc_desired()
                                        && (process.flags.intersects(
                                            ProcessFlags::DELAY_GC | ProcessFlags::DISABLE_GC,
                                        ))
                                    {
                                        trace!(target: "scheduler", "process desires garbage collection");
                                        let cost =
                                            process.garbage_collect(Default::default()).unwrap();
                                        process.reductions += cost;
                                    }

                                    // schedule out..
                                    trace!(target: "scheduler", "process is suspending");
                                    break 'schedule;
                                }

                                new_status = status;
                                new_status &= !(StatusFlags::RUNNING_SYS);
                                new_status |= StatusFlags::RUNNING;

                                match process.cmpxchg_status_flags(new_status, status) {
                                    Ok(current) => {
                                        status = current;
                                        break;
                                    }
                                    Err(current) => {
                                        status = current;
                                    }
                                }

                                assert!(status.contains(StatusFlags::RUNNING_SYS));
                                assert!(!status.contains(StatusFlags::RUNNING));
                            }
                        }

                        if process.is_gc_desired() {
                            trace!(target: "scheduler", "process desires garbage collection");
                            let flags = process.flags;
                            if !status.contains(StatusFlags::EXITING)
                                && !flags
                                    .intersects(ProcessFlags::DELAY_GC | ProcessFlags::DISABLE_GC)
                            {
                                let cost = process.garbage_collect(Default::default()).unwrap();
                                process.reductions += cost;
                                if process.reductions > MAX_REDUCTIONS {
                                    // schedule out..
                                    trace!(target: "scheduler", "process has consumed its reduction budget, yielding..");
                                    break 'schedule;
                                }
                            }
                        }

                        if status.contains(StatusFlags::EXITING) {
                            process.clear_timer_on_timeout();
                        }

                        // Never run a suspended process
                        assert!(!status.contains(StatusFlags::SUSPENDED));

                        // We're scheduled in, begin executing process
                        self.process_main(&mut process)?;
                        break 'schedule;
                    }

                    // Scheduling out..

                    // Clear active-sys if needed
                    let mut status = process.status(Ordering::Acquire);
                    trace!(target: "scheduler", "process {} is being scheduled out, current status is {:?}", process.pid(), status);
                    loop {
                        let mut new_status = status;
                        if status.contains(StatusFlags::ACTIVE_SYS) {
                            if status.intersects(
                                StatusFlags::SYS_TASKS
                                    | StatusFlags::HAS_IN_TRANSIT_SIGNALS
                                    | StatusFlags::HAS_PENDING_SIGNALS,
                            ) {
                                break;
                            }
                            new_status.remove(StatusFlags::ACTIVE_SYS);
                            match process.cmpxchg_status_flags(status, new_status) {
                                Ok(current) => {
                                    status = current;
                                    break;
                                }
                                Err(current) => {
                                    status = current;
                                }
                            }
                        } else {
                            break;
                        }
                    }

                    let running_status = StatusFlags::RUNNING | StatusFlags::RUNNING_SYS;
                    let mut enqueue;

                    loop {
                        let mut new_status = status;
                        enqueue = false;

                        assert!(status.intersects(running_status));
                        let exiting_and_free = status & (StatusFlags::EXITING | StatusFlags::FREE)
                            != StatusFlags::EXITING;
                        let active_but_not_suspended = status
                            & (StatusFlags::ACTIVE | StatusFlags::SUSPENDED)
                            == StatusFlags::ACTIVE;
                        assert!(exiting_and_free || active_but_not_suspended);

                        new_status.remove(running_status);
                        if status.contains(StatusFlags::ACTIVE_SYS) || active_but_not_suspended {
                            enqueue = true;
                        }
                        match process.cmpxchg_status_flags(status, new_status) {
                            Ok(current) => {
                                status = current;
                                break;
                            }
                            Err(current) => {
                                status = current;
                            }
                        }
                    }

                    if enqueue {
                        let is_not_suspended = !status.contains(StatusFlags::SUSPENDED);
                        let is_active_sys = status.contains(StatusFlags::ACTIVE_SYS);
                        assert!(is_not_suspended || is_active_sys);
                        trace!(target: "scheduler", "process is being requeued with status {:?}", status);
                        self.runq.push(process);
                    } else {
                        trace!(target: "scheduler", "process is not being requeued, status is {:?}", status);

                        if status.contains(StatusFlags::FREE) && process.is_init() {
                            trace!(target: "scheduler", "the init process has terminated, shutting down");
                            return Err(EmulatorError::Halt(0));
                        }
                    }

                    // Tick the timer service
                    {
                        trace!(target: "scheduler", "ticking timer wheel");
                        self.timers.borrow_mut().tick();
                    }

                    // TODO: Handle other auxiliary work on a periodic basis, say every 2 *
                    // MAX_REDUCTIONS Things include timers (handled above),
                    // ports, async tasks, etc.

                    // We return true to indicate we are ready to resume immediately
                    return Ok(true);
                }
                None => {
                    // Tick the timer service
                    let mut timers = self.timers.borrow_mut();
                    if timers.is_empty() {
                        trace!(target: "scheduler", "there are no processes to schedule, and no timers, shutting down");
                        return Err(EmulatorError::Halt(0));
                    }
                    trace!(target: "scheduler", "ticking timer wheel");
                    return Ok(timers.tick());
                }
            }
        }

        Ok(false)
    }

    /// Execute a process until:
    ///
    /// * It consumes its reduction budget, forcing it to yield
    /// * It voluntarily yields
    /// * It is suspended due to a blocking operation (e.g. `receive`)
    /// * It exits because it ran out of code to run
    /// * It exits abnormally due to an exception
    /// * It exits because it killed itself or was killed by another process
    /// * A system halt is requested
    #[inline(never)]
    fn process_main(&self, process: &mut ProcessLock) -> Result<(), EmulatorError> {
        // Resume executing user code in this process
        let mut reductions = process.reductions;
        trace!(target: "scheduler", "starting to execute process {}", process.pid());
        let mut init_op;
        loop {
            // Load current opcode, and bump instruction pointer
            let op = {
                let current_ip = process.ip;
                // Handle the case where a process is spawned with an MFA
                if likely(current_ip > 0) {
                    trace!(target: "process", "ip: {}, reductions {}", current_ip, reductions);
                    process.ip += 1;
                    &self.code.code[current_ip]
                } else {
                    let mfa = process.initial_call().into();
                    match self.code.function_by_mfa(&mfa) {
                        Some(Function::Bytecode {
                            is_nif: false,
                            offset,
                            ..
                        }) => {
                            let offset = *offset;
                            trace!(target: "process", "ip: {}, reductions {}", offset, reductions);
                            process.ip = offset + 1;
                            &self.code.code[offset as usize]
                        }
                        Some(fun) => {
                            process.ip = NORMAL_EXIT_IP;
                            init_op = Opcode::CallStatic(ops::CallStatic {
                                dest: RETURN_REG,
                                callee: fun.id(),
                            });
                            &init_op
                        }
                        None => {
                            // Check if this is a call to erlang:apply/2 or /3
                            process.ip = NORMAL_EXIT_IP;
                            if mfa.module == atoms::Erlang && mfa.function == atoms::Apply {
                                match mfa.arity {
                                    2 => {
                                        let initial_args = process.initial_arguments();
                                        match initial_args {
                                            Some(Term::Tuple(argv))
                                                if argv.len() == mfa.arity as usize =>
                                            {
                                                process.stack.store(ARG0_REG, argv[0]);
                                                process.stack.store(ARG0_REG + 1, argv[1]);
                                                init_op = Opcode::CallApply2(ops::CallApply2 {
                                                    dest: RETURN_REG,
                                                    callee: ARG0_REG,
                                                    argv: ARG0_REG + 1,
                                                });
                                                &init_op
                                            }
                                            _ => {
                                                // Raise badarg
                                                process.stack.store(
                                                    ARG0_REG + mfa.arity as Register,
                                                    atoms::Badarg.into(),
                                                );
                                                init_op = Opcode::Error1(ops::Error1 {
                                                    reason: ARG0_REG + mfa.arity as Register,
                                                });
                                                &init_op
                                            }
                                        }
                                    }
                                    3 => {
                                        let initial_args = process.initial_arguments();
                                        match initial_args {
                                            Some(Term::Tuple(argv))
                                                if argv.len() == mfa.arity as usize =>
                                            {
                                                process.stack.store(ARG0_REG, argv[0]);
                                                process.stack.store(ARG0_REG + 1, argv[1]);
                                                process.stack.store(ARG0_REG + 2, argv[2]);
                                                init_op = Opcode::CallApply3(ops::CallApply3 {
                                                    dest: RETURN_REG,
                                                    module: ARG0_REG,
                                                    function: ARG0_REG + 1,
                                                    argv: ARG0_REG + 2,
                                                });
                                                &init_op
                                            }
                                            _ => {
                                                // Raise badarg
                                                process.stack.store(
                                                    ARG0_REG + mfa.arity as Register,
                                                    atoms::Badarg.into(),
                                                );
                                                init_op = Opcode::Error1(ops::Error1 {
                                                    reason: ARG0_REG + mfa.arity as Register,
                                                });
                                                &init_op
                                            }
                                        }
                                    }
                                    _ => {
                                        process.stack.store(
                                            ARG0_REG + mfa.arity as Register,
                                            atoms::Undef.into(),
                                        );
                                        init_op = Opcode::Error1(ops::Error1 {
                                            reason: ARG0_REG + mfa.arity as Register,
                                        });
                                        &init_op
                                    }
                                }
                            } else {
                                // Raise undef
                                process
                                    .stack
                                    .store(ARG0_REG + mfa.arity as Register, atoms::Undef.into());
                                init_op = Opcode::Error1(ops::Error1 {
                                    reason: ARG0_REG + mfa.arity as Register,
                                });
                                &init_op
                            }
                        }
                    }
                }
            };

            // Dispatch on the current instruction
            match op.dispatch(self, &mut *process) {
                Action::Continue => {
                    // Check if we've exhausted our reductions this cycle
                    //
                    // Each iteration we increment the total reductions by the number
                    // consumed during the cycle.
                    reductions += process.reductions - reductions;
                    if unlikely(process.reductions >= MAX_REDUCTIONS) {
                        process.reductions = 0;
                        trace!(target: "process", "reduction budget exhausted, yielding..");
                        break;
                    }
                }
                Action::Yield => {
                    reductions += process.reductions - reductions;
                    trace!(target: "process", "yielding at {} reductions", reductions);
                    break;
                }
                Action::Killed => {
                    trace!(target: "process", "killed");
                    reductions += process.reductions - reductions;
                    break;
                }
                Action::Suspend => {
                    reductions += process.reductions - reductions;
                    process.set_status_flags(StatusFlags::SUSPENDED, Ordering::Release);
                    let status =
                        process.remove_status_flags(StatusFlags::ACTIVE, Ordering::Release);
                    trace!(target: "process", "suspending with status {:?}", status & !StatusFlags::ACTIVE);
                    break;
                }
                Action::Error(e) => return Err(e),
            }
        }

        self.reductions
            .fetch_add(reductions as u64, Ordering::Relaxed);

        Ok(())
    }

    pub(crate) fn handle_signals(
        &self,
        process: &mut ProcessLock,
        status: &mut StatusFlags,
        reductions: usize,
        mut local_only: bool,
    ) -> Action {
        trace!(target: "process", "handling signals (local_only = {})", local_only);
        let limit = reductions * ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR;
        let mut count = 0;
        let proc = process.strong();
        let mut signals = proc.signals().lock();

        if status.contains(StatusFlags::EXITING) {
            // ?
            return Action::Yield;
        }

        loop {
            count += 1;
            if count >= limit {
                break;
            }

            let signal;
            if let Some(sig) = signals.pop_signal(local_only) {
                trace!(target: "process", "handling signal {:?}", &sig.signal);
                signal = sig;
                // After the first non-local fetch, only do local fetches
                if unlikely(!local_only) {
                    local_only = true;
                }
            } else {
                trace!(target: "process", "processed all signals");
                break;
            }

            match signal.signal {
                Signal::Exit(_) | Signal::ExitLink(_) => {
                    let (cost, exited) = self.handle_exit_signal(process, &mut signals, signal);
                    count += cost;
                    // Terminated by signal
                    if exited {
                        break;
                    }
                    // Ignored or converted to message
                }
                Signal::MonitorDown(sig) => {
                    let monitor_ref = sig.monitor.key();
                    match &sig.monitor.monitor {
                        Monitor::LocalProcess { .. }
                        | Monitor::LocalPort { .. }
                        | Monitor::FromExternalProcess { .. }
                        | Monitor::ToExternalProcess { .. } => {
                            assert!(!sig.monitor.is_target_linked());
                            let reason: Term = sig.reason.term.into();
                            drop(sig.monitor);
                            if let MonitorTreeEntry::Occupied(mut cursor) =
                                process.monitored.entry(&monitor_ref)
                            {
                                let monitor = cursor.get().unwrap();
                                let message: signals::Message;
                                assert!(reason.is_immediate());
                                let flags = monitor.flags();
                                if flags.contains(MonitorFlags::SPAWN_PENDING) {
                                    // Create a spawn_request() error message and replace the signal
                                    // with it Should only
                                    // happens when connection breaks;
                                    assert_eq!(reason, atoms::Noconnection);
                                    if flags.intersects(
                                        MonitorFlags::SPAWN_ABANDONED
                                            | MonitorFlags::SPAWN_NO_REPLY_ERROR,
                                    ) {
                                        // Operation has been abandoned or error message has been
                                        // disabled..
                                        cursor.remove();
                                        continue;
                                    }
                                    count += 4;
                                    let mut layout = LayoutBuilder::new();
                                    let tag = {
                                        match monitor.tag() {
                                            None => Term::Nil,
                                            Some(t) => {
                                                let tag: Term = t.into();
                                                layout += tag.layout();
                                                tag
                                            }
                                        }
                                    };
                                    layout += reason.layout();
                                    layout.build_reference();
                                    layout.build_tuple(4);
                                    let fragment_ptr = layout.into_fragment().unwrap();
                                    let fragment = unsafe { fragment_ptr.as_ref() };
                                    let reason =
                                        unsafe { reason.unsafe_clone_to_heap(fragment).into() };
                                    let tag = unsafe { tag.unsafe_clone_to_heap(fragment).into() };
                                    let mref = Gc::new_in(monitor_ref, fragment).unwrap().into();
                                    let msg = Tuple::from_slice(
                                        &[tag, mref, atoms::Error.into(), reason],
                                        fragment,
                                    )
                                    .unwrap();
                                    message = Message {
                                        sender: WeakAddress::Name(atoms::Undefined),
                                        message: TermFragment {
                                            term: msg.into(),
                                            fragment: Some(fragment_ptr),
                                        },
                                    };

                                    // Restore to normal monitor
                                    monitor.remove_flags(MonitorFlags::SPAWN_MASK);
                                } else {
                                    // Create a DOWN message and replace the signal with it
                                    let mut layout = LayoutBuilder::new();
                                    layout += reason.layout();
                                    let mut tag;
                                    let origin;
                                    if !flags.contains(MonitorFlags::TAG) {
                                        // Registered name 2-tuple
                                        layout.build_tuple(2);
                                        origin = WeakAddress::System;
                                        tag = Term::Atom(atoms::DOWN);
                                    } else {
                                        match monitor.origin().unwrap_or(WeakAddress::System) {
                                            addr @ WeakAddress::Process(_) => {
                                                origin = addr;
                                                layout.build_pid();
                                            }
                                            addr @ WeakAddress::Port(_) => {
                                                origin = addr;
                                                layout.build_port();
                                            }
                                            addr @ WeakAddress::System => {
                                                origin = addr;
                                            }
                                            _ => panic!("expected pid or port"),
                                        }
                                        if let Some(t) = monitor.tag() {
                                            tag = t.into();
                                            layout += tag.layout();
                                        } else {
                                            tag = Term::Atom(atoms::DOWN);
                                        }
                                    }
                                    layout.build_reference();
                                    layout.build_tuple(5);
                                    let fragment_ptr = layout.into_fragment().unwrap();
                                    let fragment = unsafe { fragment_ptr.as_ref() };
                                    let reason =
                                        unsafe { reason.unsafe_clone_to_heap(fragment).into() };

                                    let from;
                                    let ty;
                                    if !flags.contains(MonitorFlags::TAG) {
                                        let name = monitor.name().unwrap();
                                        let node = monitor.node_name();
                                        from = Term::Tuple(
                                            Tuple::from_slice(
                                                &[name.into(), node.into()],
                                                fragment,
                                            )
                                            .unwrap(),
                                        );
                                    } else {
                                        match origin {
                                            WeakAddress::Process(ref pid) => {
                                                from = Term::Pid(
                                                    Gc::new_in(pid.clone(), fragment).unwrap(),
                                                );
                                            }
                                            WeakAddress::Port(port) => {
                                                from = match registry::get_by_port_id(port) {
                                                    None => Term::Atom(atoms::System),
                                                    Some(p) => Term::Port(p),
                                                };
                                            }
                                            WeakAddress::System => {
                                                from = Term::Atom(atoms::System);
                                            }
                                            _ => unreachable!(),
                                        }
                                    }
                                    match &monitor.monitor {
                                        Monitor::LocalPort { .. } => {
                                            ty = atoms::Port;
                                        }
                                        Monitor::LocalProcess { .. }
                                        | Monitor::ToExternalProcess { .. } => {
                                            ty = atoms::Process;
                                        }
                                        Monitor::FromExternalProcess { .. } => {
                                            ty = atoms::Process;
                                        }
                                        _ => panic!("unexpected monitor type"),
                                    }
                                    tag = unsafe { tag.unsafe_clone_to_heap(fragment) };
                                    let mref = Gc::new_in(monitor_ref.clone(), fragment).unwrap();
                                    let term = Tuple::from_slice(
                                        &[tag.into(), mref.into(), ty.into(), from.into(), reason],
                                        fragment,
                                    )
                                    .unwrap();
                                    message = Message {
                                        sender: origin,
                                        message: TermFragment {
                                            term: term.into(),
                                            fragment: Some(fragment_ptr),
                                        },
                                    };
                                }
                                count += 4;
                                unsafe {
                                    signals.push_next_message(SignalEntry::new(Signal::Message(
                                        message,
                                    )));
                                }
                            }
                        }
                        Monitor::Node { .. } => {
                            //handle_nodedown(process, sig);
                            todo!()
                        }
                        Monitor::Suspend { .. } => {
                            let mut cursor = process.monitored.find_mut(&monitor_ref);
                            cursor.remove();
                        }
                        _ => panic!("invalid monitor type"),
                    }
                }
                Signal::Monitor(sig) => {
                    count += self.handle_monitor(process, sig.monitor);
                }
                Signal::Demonitor(sig) => {
                    assert!(!sig.monitor.is_origin_linked());
                    assert_eq!(sig.monitor.target(), Some(process.addr()));
                    match &sig.monitor.monitor {
                        Monitor::FromExternalProcess { .. } => {
                            // TODO: destroy_dist_proc_demonitor
                            if sig.monitor.is_target_linked() {
                                let mut cursor = unsafe { process.monitored_by.cursor_mut_from_ptr(Arc::as_ptr(&sig.monitor)) };
                                cursor.remove();
                                count += 2;
                            }
                            count += 1;
                        }
                        Monitor::Resource { .. } => {
                            // erts_nif_demonitored(resource);
                            // let mut cursor = unsafe { process.monitored_by.cursor_mut_from_ptr(Arc::as_ptr(&sig.monitor)) };
                            // cursor.remove();
                            // count += 1;
                            todo!()
                        }
                        Monitor::Suspend { /* ref info, */ .. } => {
                            // let mut cursor = unsafe { process.monitored_by.cursor_mut_from_ptr(Arc::as_ptr(&sig.monitor)) };
                            // cursor.remove();
                            // if info.active {
                            //     erts_resume(process);
                            // }
                            todo!()
                        }
                        _ => {
                            if sig.monitor.is_target_linked() {
                                let mut cursor = unsafe { process.monitored_by.cursor_mut_from_ptr(Arc::as_ptr(&sig.monitor)) };
                                cursor.remove();
                                count += 1;
                            }
                        }
                    }
                    count += 1;
                }
                Signal::Link(sig) => {
                    if let Err(entry) = process.links.linked_by(sig.link) {
                        // Already linked or unlinking
                        if let Link::FromExternalProcess { .. } = entry.link {
                            // Need to remove the new link from distribution
                            todo!()
                        }
                    }
                }
                Signal::Unlink(sig) => {
                    count += self.handle_unlink(process, sig.sender, sig.id);
                }
                Signal::UnlinkAck(sig) => {
                    if let LinkTreeEntry::Occupied(entry) = process.links.entry(&sig.sender) {
                        let is_unlinking = entry.get().unlinking() == Some(sig.id);
                        if is_unlinking {
                            let link_entry = entry.remove();
                            match link_entry.link {
                                Link::LocalProcess { .. } | Link::LocalPort { .. } => {
                                    count += 4;
                                }
                                Link::ToExternalProcess { .. }
                                | Link::FromExternalProcess { .. } => {
                                    count += 8;
                                }
                            }
                        }
                    }
                }
                Signal::GroupLeader(sig) => {
                    let update = !status.contains(StatusFlags::EXITING);
                    if update {
                        process.set_group_leader(sig.group_leader);
                    }
                    let sender = WeakAddress::Process(sig.sender);
                    if let Some(Registrant::Process(p)) = sender.try_resolve() {
                        self.send_group_leader_reply(p, sig.reference, update);
                    }
                }
                Signal::IsAlive(sig) => {
                    let sender = WeakAddress::Process(sig.sender);
                    if let Some(Registrant::Process(p)) = sender.try_resolve() {
                        self.send_is_alive_reply(p, sig.reference, true);
                    }
                }
                Signal::ProcessInfo(sig) => {
                    let is_alive = !status.contains(StatusFlags::EXITING);
                    self.handle_process_info_signal(process, sig, is_alive);
                }
                Signal::Rpc(sig) => {
                    count += self.handle_rpc(process, sig);
                }
                Signal::Message(_) | Signal::Flush(_) => unreachable!(),
            }
        }

        process.reductions += 1 + (count / ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR);
        if signals.has_pending_signals() {
            *status = process.status(Ordering::Relaxed);
        } else {
            *status =
                process.remove_status_flags(StatusFlags::HAS_PENDING_SIGNALS, Ordering::Relaxed);
            *status &= !StatusFlags::HAS_PENDING_SIGNALS;
        }

        if status.contains(StatusFlags::MAYBE_SELF_SIGNALS)
            && !status
                .intersects(StatusFlags::HAS_PENDING_SIGNALS | StatusFlags::HAS_IN_TRANSIT_SIGNALS)
        {
            // We know we do not have any outstanding signals from ourselves
            *status =
                process.remove_status_flags(StatusFlags::MAYBE_SELF_SIGNALS, Ordering::Relaxed);
            *status &= StatusFlags::MAYBE_SELF_SIGNALS;
        }

        if process.reductions >= MAX_REDUCTIONS {
            Action::Yield
        } else {
            Action::Continue
        }
    }

    fn handle_monitor(&self, process: &mut ProcessLock, monitor: Arc<MonitorEntry>) -> usize {
        match &monitor.monitor {
            Monitor::Suspend { .. } => {
                // handle_suspend
                // process.monitored_by.push_back(sig.monitor);
                todo!()
            }
            _ => {
                process.monitored_by.push_back(monitor);
            }
        }

        2
    }

    // See erts_proc_handle_exit_link
    fn handle_exit_link(
        &self,
        process: &mut ProcessLock,
        reason: OpaqueTerm,
        link: Arc<LinkEntry>,
    ) {
        let exiting = process.id();
        match &link.link {
            Link::LocalProcess { origin, target } if exiting == *origin => {
                if let Some(target) = registry::get_by_process_id(*target) {
                    target
                        .send_signal(SignalEntry::new(Signal::ExitLink(signals::Exit {
                            sender: Some(process.addr()),
                            reason: TermFragment::new(reason.into()).unwrap(),
                            normal_kills: false,
                        })))
                        .ok();
                }
            }
            Link::LocalProcess { origin, target } => {
                assert_eq!(exiting, *target);
                if let Some(origin) = registry::get_by_process_id(*origin) {
                    origin
                        .send_signal(SignalEntry::new(Signal::ExitLink(signals::Exit {
                            sender: Some(process.addr()),
                            reason: TermFragment::new(reason.into()).unwrap(),
                            normal_kills: false,
                        })))
                        .ok();
                }
            }
            _ => unimplemented!(),
        }
    }

    // See erts_proc_handle_exit_monitor
    //
    // This is called for monitors targeting `process`, not monitors originated by `process`
    fn handle_exit_monitor(
        &self,
        process: &mut ProcessLock,
        reason: OpaqueTerm,
        monitor: Arc<MonitorEntry>,
    ) {
        match &monitor.monitor {
            Monitor::Suspend { origin, .. } | Monitor::LocalProcess { origin, .. } => {
                let reason = if let Monitor::Suspend { .. } = &monitor.monitor {
                    atoms::Undefined.into()
                } else {
                    reason
                };
                if let Some(origin) = registry::get_by_process_id(*origin) {
                    origin
                        .send_signal(SignalEntry::new(Signal::MonitorDown(
                            signals::MonitorDown {
                                sender: Some(process.addr()),
                                reason: TermFragment::new(reason.into()).unwrap(),
                                monitor,
                            },
                        )))
                        .ok();
                }
            }
            _ => unimplemented!(),
        }
    }

    // See erts_proc_handle_exit_monitor
    //
    // This is called for monitors originated by `process`, not monitors targeting `process`
    fn handle_exit_demonitor(
        &self,
        process: &mut ProcessLock,
        _reason: OpaqueTerm,
        monitor: Arc<MonitorEntry>,
    ) {
        match &monitor.monitor {
            Monitor::Suspend { origin, .. } | Monitor::LocalProcess { origin, .. } => {
                if let Some(origin) = registry::get_by_process_id(*origin) {
                    origin
                        .send_signal(SignalEntry::new(Signal::Demonitor(signals::Demonitor {
                            sender: process.addr(),
                            monitor,
                        })))
                        .ok();
                }
            }
            _ => unimplemented!(),
        }
    }

    fn handle_unlink(
        &self,
        process: &mut ProcessLock,
        sender: WeakAddress,
        id: NonZeroU64,
    ) -> usize {
        let from = process.addr();
        match process.links.entry(&sender) {
            LinkTreeEntry::Vacant(_) => {
                self.send_unlink_ack(from, sender, id);
                1
            }
            LinkTreeEntry::Occupied(entry) => {
                let is_unlinking = entry.get().unlinking().contains(&id);
                if is_unlinking {
                    let link_entry = entry.remove();
                    match link_entry.link {
                        Link::LocalProcess { .. } | Link::LocalPort { .. } => {
                            self.send_unlink_ack(from, sender, id);
                            4
                        }
                        Link::ToExternalProcess { .. } | Link::FromExternalProcess { .. } => {
                            // count += 8;
                            // self.send_dist_unlink_ack(process.id().into(), sender, id);
                            todo!()
                        }
                    }
                } else {
                    let link_entry = entry.get();
                    match link_entry.link {
                        Link::LocalProcess { .. } | Link::LocalPort { .. } => {
                            self.send_unlink_ack(from, sender, id);
                            1
                        }
                        Link::ToExternalProcess { .. } | Link::FromExternalProcess { .. } => {
                            // self.send_dist_unlink_ack(process.id().into(), sender, id);
                            // 1
                            todo!()
                        }
                    }
                }
            }
        }
    }

    fn send_unlink_ack(&self, from: WeakAddress, to: WeakAddress, id: NonZeroU64) {
        if let Some(registrant) = to.try_resolve() {
            match registrant {
                Registrant::Process(proc) => {
                    proc.send_signal(SignalEntry::new(Signal::UnlinkAck(signals::UnlinkAck {
                        sender: from,
                        id,
                    })))
                    .ok();
                }
                _ => unimplemented!(),
            }
        }
    }

    fn handle_exit_signal(
        &self,
        process: &mut ProcessLock,
        signals: &mut SignalQueueLock<'_>,
        mut signal: Box<SignalEntry>,
    ) -> (usize, bool) {
        let mut ignore = false;
        let mut is_link_exit = false;
        let mut exit = false;
        let mut count = 1;
        let sender;
        let reason_fragment: Option<NonNull<HeapFragment>>;
        let mut reason: OpaqueTerm;
        let normal_kills;
        match signal.signal {
            Signal::ExitLink(sig) => {
                sender = sig.sender.unwrap();
                if let Some(entry) = process.links.unlink(&sender) {
                    if entry.unlinking().is_some() {
                        ignore = true;
                    }
                } else {
                    ignore = true;
                }
                normal_kills = sig.normal_kills;
                reason = sig.reason.term;
                reason_fragment = sig.reason.fragment;
                is_link_exit = true;
            }
            Signal::Exit(sig) => {
                sender = sig.sender.unwrap();
                normal_kills = sig.normal_kills;
                reason = sig.reason.term;
                reason_fragment = sig.reason.fragment;
            }
            _ => unreachable!(),
        }

        if !ignore {
            if (is_link_exit || reason != atoms::Kill)
                && process.flags.contains(ProcessFlags::TRAP_EXIT)
            {
                signal.signal = Signal::Message(signals::Message {
                    sender,
                    message: TermFragment {
                        term: reason.into(),
                        fragment: reason_fragment,
                    },
                });
                assert!(!exit);
                unsafe {
                    signals.push_next_message(signal);
                }
            } else if !(reason == atoms::Normal && !normal_kills) {
                // terminate
                exit = true;
                if !is_link_exit && reason == atoms::Kill {
                    reason = atoms::Killed.into();
                }
            }
        }

        if exit {
            // set_self_exiting
            if let Some(ptr) = reason_fragment {
                process
                    .heap_fragments
                    .push_back(unsafe { UnsafeRef::from_raw(ptr.as_ptr().cast_const()) });
            }
            process.exception_info.value = reason.into();
            process.exception_info.flags = ExceptionFlags::EXIT;
            process.exception_info.trace = None;
            process.stack.nocatch();
            process.ip = CONTINUE_EXIT_IP;
            process.set_status_flags(
                StatusFlags::EXITING | StatusFlags::ACTIVE,
                Ordering::Release,
            );
            process.remove_status_flags(StatusFlags::SUSPENDED, Ordering::Relaxed);

            count += 1;
        }

        (count, exit)
    }

    fn handle_process_info_signal(
        &self,
        _process: &mut ProcessLock,
        _sig: signals::ProcessInfo,
        _is_alive: bool,
    ) {
        todo!()
    }

    fn handle_rpc(&self, process: &mut ProcessLock, sig: signals::Rpc) -> usize {
        let in_reds = process.reductions;
        let result = (sig.callback)(process, sig.arg);
        let cost = process.reductions.saturating_sub(in_reds);
        process.reductions = in_reds;
        if let Some(reply_ref) = sig.reference {
            // Reply requested
            let requestor = registry::get_by_pid(&sig.sender);
            if let Some(requestor) = requestor {
                let mut layout = LayoutBuilder::new();
                layout += Layout::new::<Reference>();
                let result_term: Term = result.term.into();
                layout += result_term.layout();
                let fragment_ptr = HeapFragment::new(layout.finish(), None).unwrap();
                let fragment = unsafe { fragment_ptr.as_ref() };
                let reply_ref = Gc::new_in(reply_ref, fragment).unwrap();
                let result = unsafe { result_term.unsafe_clone_to_heap(fragment) };
                let msg = Tuple::from_slice(&[reply_ref.into(), result.into()], fragment).unwrap();
                requestor
                    .send_fragment(
                        process.pid().into(),
                        TermFragment {
                            term: msg.into(),
                            fragment: Some(fragment_ptr),
                        },
                    )
                    .ok();
            }
        }
        cost * ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR
    }

    fn send_group_leader_reply(&self, to: Arc<Process>, reference: Reference, success: bool) {
        let mut layout = LayoutBuilder::new();
        layout.build_reference().build_tuple(2);
        let fragment_ptr = layout.into_fragment().unwrap();
        let fragment = unsafe { fragment_ptr.as_ref() };

        let glref = Gc::new_in(reference, fragment).unwrap();
        let result = if success {
            true.into()
        } else {
            atoms::Badarg.into()
        };
        let tuple = Tuple::from_slice(&[glref.into(), result], fragment).unwrap();

        to.send_fragment(
            WeakAddress::System,
            TermFragment {
                term: tuple.into(),
                fragment: Some(fragment_ptr),
            },
        )
        .ok();
    }

    fn send_is_alive_reply(&self, to: Arc<Process>, reference: Reference, is_alive: bool) {
        let mut layout = LayoutBuilder::new();
        layout.build_reference().build_tuple(2);
        let fragment_ptr = layout.into_fragment().unwrap();
        let fragment = unsafe { fragment_ptr.as_ref() };

        let glref = Gc::new_in(reference, fragment).unwrap();
        let tuple = Tuple::from_slice(&[glref.into(), is_alive.into()], fragment).unwrap();

        to.send_fragment(
            WeakAddress::System,
            TermFragment {
                term: tuple.into(),
                fragment: Some(fragment_ptr),
            },
        )
        .ok();
    }

    fn execute_sys_tasks(&self, process: &mut ProcessLock, statusp: &mut StatusFlags) -> usize {
        use firefly_rt::process::{Priority, SystemTaskType};

        trace!(target: "process", "executing system tasks");

        let mut priority = Priority::Max as usize;
        let mut gc_major = false;
        let mut gc_minor = false;
        let mut reds = process.reductions;
        let mut status = *statusp;
        loop {
            if status.contains(StatusFlags::EXITING) {
                break;
            }

            if reds == 0 {
                break;
            }

            let next = process.system_tasks[priority].pop_front();
            if next.is_none() {
                if priority == 0 {
                    break;
                }
                priority -= 1;
                continue;
            }

            let task = next.unwrap();
            let task_result: OpaqueTerm = match task.ty {
                ty @ (SystemTaskType::GcMajor | SystemTaskType::GcMinor) => {
                    if process.flags.contains(ProcessFlags::DISABLE_GC) {
                        // reds -= 1;
                        todo!("save_gc_task");
                    }
                    let is_major = ty == SystemTaskType::GcMajor;
                    if (!gc_minor || (!gc_major && is_major))
                        && !process.flags.contains(ProcessFlags::HIBERNATED)
                    {
                        if is_major {
                            process.flags |= ProcessFlags::NEED_FULLSWEEP;
                        }
                        let cost = process.garbage_collect(Default::default()).unwrap();
                        reds = reds.saturating_sub(cost);
                        gc_minor = true;
                        gc_major = gc_major || is_major;
                    }
                    true.into()
                }
                SystemTaskType::Test => true.into(),
            };

            crate::bifs::erlang::notify_sys_task_executed(process, task, task_result);
            reds = reds.saturating_sub(1);

            status = process.status(Ordering::Acquire);
        }

        *statusp = status;

        process.reductions.saturating_sub(reds)
    }

    fn cleanup_sys_tasks(&self, process: &mut ProcessLock) {
        use firefly_rt::process::{Priority, SystemTaskType};

        let mut priority = Priority::Max as usize;
        let mut reds = process.reductions;
        loop {
            if reds == 0 {
                break;
            }

            let next = process.system_tasks[priority].pop_front();
            if next.is_none() {
                if priority == 0 {
                    break;
                }
                priority -= 1;
                continue;
            }

            let task = next.unwrap();
            let task_result: OpaqueTerm = match task.ty {
                SystemTaskType::GcMajor | SystemTaskType::GcMinor | SystemTaskType::Test => {
                    false.into()
                }
            };

            crate::bifs::erlang::notify_sys_task_executed(process, task, task_result);
            reds = reds.saturating_sub(1);
        }
    }

    /// Register a timeout for the given process
    fn timeout_after(
        &self,
        process: &mut ProcessLock,
        timeout: Timeout,
    ) -> Result<ReferenceId, TimerError> {
        let timer_ref = self.next_reference_id();
        self.timers
            .borrow_mut()
            .timeout_after(timer_ref, process, timeout)
            .map(|_| timer_ref)
    }

    /// Generate a stack trace for the current process
    fn get_stacktrace(&self, process: &mut ProcessLock) -> Arc<Trace> {
        use firefly_rt::backtrace::Frame;
        /*
        let initial_mfa = process.initial_call().into();
        let inital_frame = {
            let initial_fun = self.code.function_by_mfa(&initial_mfa);
            let initial_symbol = self.code.function_symbol(initial_fun.id());
            let frame: Box<dyn Frame> = Box::new(initial_symbol);
            TraceFrame::from(frame)
        };
        */
        let frames = core::iter::once(process.ip)
            .chain(process.stack.trace(None).map(|f| f.ret))
            .filter_map(|ip| {
                if ip == 0 {
                    None
                } else {
                    let function = self.code.function_by_ip(ip);
                    let symbol = self.code.function_symbol(function.id());
                    let frame: Box<dyn Frame> = Box::new(symbol);
                    Some(TraceFrame::from(frame))
                }
            })
            .collect();

        Trace::new(frames)
    }

    fn parse_stacktrace(&self, framelist: Gc<Cons>) -> Result<Arc<Trace>, ()> {
        use firefly_rt::backtrace::{Frame, FrameWithExtraInfo};

        let mut frames = Vec::with_capacity(10);
        // Reconstruct a TraceFrame from each element in the frame list
        for frame in framelist.iter() {
            // Stack traces must be proper lists, with 2-,3-,or 4-tuple elements.
            match frame.map_err(|_| ())? {
                Term::Tuple(tuple) => {
                    let module: Atom;
                    let function: Atom;
                    let arity: u8;
                    let args: Term;
                    let extra_info: Term;
                    match tuple.len() {
                        4 => {
                            // {Module, Function, Arity | Args, ExtraInfo}
                            module = match tuple[0].into() {
                                Term::Atom(m) => m,
                                _ => return Err(()),
                            };
                            function = match tuple[1].into() {
                                Term::Atom(f) => f,
                                _ => return Err(()),
                            };
                            match tuple[2].into() {
                                Term::Int(a) => {
                                    arity = a.try_into().map_err(|_| ())?;
                                    args = Term::Nil;
                                }
                                Term::Nil => {
                                    arity = 0;
                                    args = Term::Nil;
                                }
                                Term::Cons(cons) => {
                                    let mut a = 0;
                                    for maybe in cons.iter_raw() {
                                        if maybe.is_err() {
                                            return Err(());
                                        }
                                        a += 1;
                                    }
                                    arity = a;
                                    args = Term::Cons(cons);
                                }
                                _ => return Err(()),
                            }
                            extra_info = match tuple[3].into() {
                                extra @ (Term::Nil | Term::Cons(_)) => extra,
                                _ => return Err(()),
                            };
                        }
                        3 => {
                            // {Module, Function, Arity | Args} | {Fun, Args, ExtraInfo}
                            if tuple[0].is_atom() {
                                module = tuple[0].as_atom();
                                if !tuple[1].is_atom() {
                                    return Err(());
                                }
                                function = tuple[1].as_atom();
                                match tuple[2].into() {
                                    Term::Int(i) => {
                                        arity = i.try_into().map_err(|_| ())?;
                                        args = Term::Nil;
                                    }
                                    Term::Nil => {
                                        arity = 0;
                                        args = Term::Nil;
                                    }
                                    Term::Cons(cons) => {
                                        let mut a = 0;
                                        for arg in cons.iter_raw() {
                                            if arg.is_err() {
                                                return Err(());
                                            }
                                            a += 1;
                                        }
                                        arity = a;
                                        args = Term::Cons(cons);
                                    }
                                    _ => return Err(()),
                                }
                                extra_info = Term::Nil;
                            } else {
                                if let Term::Closure(fun) = tuple[0].into() {
                                    let mfa = fun.mfa();
                                    module = mfa.module;
                                    function = mfa.function;
                                    match tuple[1].into() {
                                        argv @ (Term::Nil | Term::Cons(_)) => {
                                            arity = mfa.arity;
                                            args = argv;
                                        }
                                        Term::Int(i) => {
                                            arity = i.try_into().map_err(|_| ())?;
                                            args = Term::Nil;
                                        }
                                        _ => return Err(()),
                                    }
                                } else {
                                    return Err(());
                                }
                                extra_info = match tuple[2].into() {
                                    extra @ (Term::Nil | Term::Cons(_)) => extra,
                                    _ => return Err(()),
                                };
                            }
                        }
                        2 => {
                            // {Fun, Arity | Args}
                            if let Term::Closure(fun) = tuple[0].into() {
                                let mfa = fun.mfa();
                                module = mfa.module;
                                function = mfa.function;
                                match tuple[1].into() {
                                    argv @ (Term::Nil | Term::Cons(_)) => {
                                        arity = mfa.arity;
                                        args = argv;
                                    }
                                    Term::Int(i) => {
                                        arity = i.try_into().map_err(|_| ())?;
                                        args = Term::Nil;
                                    }
                                    _ => return Err(()),
                                }
                                extra_info = Term::Nil;
                            } else {
                                return Err(());
                            }
                        }
                        _ => return Err(()),
                    }

                    let mfa = bc::ModuleFunctionArity {
                        module,
                        function,
                        arity,
                    };

                    // Try to parse location information from the trace's extra info
                    let mut files: HashSet<Rc<str>> = HashSet::default();
                    let mut line: Option<u32> = None;
                    let mut column: Option<u32> = None;
                    let mut file: Option<Rc<str>> = None;
                    if let Term::Cons(info) = extra_info {
                        for item in info.iter() {
                            if let Term::Tuple(meta) = item.map_err(|_| ())? {
                                if meta.len() != 2 || !meta[0].is_atom() {
                                    return Err(());
                                }
                                let key = meta[0].as_atom();
                                if key == atoms::Line {
                                    if let Term::Int(ln) = tuple[1].into() {
                                        line = Some(ln.try_into().map_err(|_| ())?);
                                        continue;
                                    }
                                } else if key == atoms::Column {
                                    if let Term::Int(cn) = tuple[1].into() {
                                        column = Some(cn.try_into().map_err(|_| ())?);
                                        continue;
                                    }
                                } else if key == atoms::File {
                                    match tuple[1].into() {
                                        Term::Nil => {
                                            file = None;
                                            continue;
                                        }
                                        Term::Cons(chardata) => {
                                            if let Some(f) = chardata.as_ref().to_string() {
                                                file = match files.get(f.as_str()) {
                                                    None => {
                                                        let file: Rc<str> = f.into();
                                                        files.insert(file.clone());
                                                        Some(file)
                                                    }
                                                    Some(f) => Some(f.clone()),
                                                };
                                            }
                                            continue;
                                        }
                                        _ => return Err(()),
                                    }
                                } else {
                                    continue;
                                }
                            }
                            return Err(());
                        }
                    }

                    // Finally, reconstruct the Symbol using what data we have.
                    //
                    // If no extra info was present, use the source location of the function, if
                    // present
                    //
                    // Erlang traces with location info always have at least a line number, so we
                    // use that as a signal that we have location info in the
                    // trace.
                    let symbol = match line {
                        None => {
                            let symbol = self
                                .code
                                .function_by_mfa(&mfa)
                                .map(|f| self.code.function_symbol(f.id()));
                            match symbol {
                                None => bc::Symbol::Erlang { mfa, loc: None },
                                Some(sym) => sym,
                            }
                        }
                        Some(line) => bc::Symbol::Erlang {
                            mfa,
                            loc: Some(bc::SourceLocation {
                                file: file.unwrap_or_else(|| Rc::from("empty")),
                                line,
                                column: column.unwrap_or(0),
                            }),
                        },
                    };
                    let frame: Box<dyn Frame> = FrameWithExtraInfo::new(symbol, args.into());
                    frames.push(TraceFrame::from(frame))
                }
                _other => return Err(()),
            }
        }

        Ok(Trace::new_with_term(frames, Term::Cons(framelist)))
    }

    fn handle_error(&self, process: &mut ProcessLock) -> Action {
        assert_ne!(process.exception_info.reason, ErrorCode::Other(atoms::Trap));
        trace!(target: "process", "handling error: {:?}", &process.exception_info);

        let mut flags = process.exception_info.flags;

        if flags.contains(ExceptionFlags::RESTORE_NIF) {
            todo!()
        }

        // Check if we have an arglist and possibly extended error info term
        // for the top-level call. If so, this is encoded in the exception reason,
        // so we have to extract the real reason + the extra metadata
        let mut value = process.exception_info.value;
        if flags.contains(ExceptionFlags::ARGS) {
            match value.into() {
                Term::Tuple(tuple) => {
                    value = tuple[0];
                    process.exception_info.args = Some(tuple[1]);
                    if tuple.len() == 3 {
                        // Dig out the `error_info` term passed to error/3
                        assert!(flags.contains(ExceptionFlags::EXTENDED));
                        process.exception_info.value = tuple[2];
                    }
                }
                _ => unreachable!(),
            }
        }

        // Save existing stack trace info if this flag is set.
        //
        // The main reason for doing this separately is to allow throws
        // to later be promoted to errors without losing the original stack
        // trace, even if they pass through one or more catch/rethrows.
        if flags.contains(ExceptionFlags::SAVETRACE) {
            if process.exception_info.trace.is_none() {
                process.exception_info.trace = Some(self.get_stacktrace(process));
            }
        } else {
            process.exception_info.trace = Some(self.get_stacktrace(process));
        }

        // Throws that are not caught are turned into 'nocatch' errors
        if flags.contains(ExceptionFlags::THROWN) && !process.stack.catches() {
            value = Tuple::from_slice(&[atoms::Nocatch.into(), value], process)
                .unwrap()
                .into();
            flags = ExceptionFlags::ERROR;
            process.exception_info.flags = ExceptionFlags::ERROR;
        }

        // Expand the error value
        value = match process.exception_info.reason {
            ErrorCode::Primary(_) => value,
            ErrorCode::Other(reason) => reason.into(),
            code => {
                // Other common exceptions are expanded from `Value` to `{Atom, Value}`
                let atom: Atom = code.into();
                Tuple::from_slice(&[atom.into(), value], process)
                    .unwrap()
                    .into()
            }
        };

        // Stablize the exception flags from this point onward
        flags = flags.to_primary();
        process.exception_info.flags = flags;
        process.exception_info.reason = process.exception_info.reason.to_primary();
        process.exception_info.value = value;

        // Find a handler or die
        if !flags.contains(ExceptionFlags::PANIC) {
            if let Some(ip) = process.stack.unwind() {
                trace!(target: "process", "exception unwound to catch handler at offset {}", ip);
                process.ip = ip;
                return Action::Continue;
            }
        }

        trace!(target: "process", "exception was uncaught, terminating process");

        self.terminate_process(process, value)
    }

    // see beam_common.c:681
    fn terminate_process(&self, process: &mut ProcessLock, reason: OpaqueTerm) -> Action {
        const EXIT_FORMAT: &'static str = "Error in process ~p with exit value:~n~p~n";
        trace!(target: "process", "terminating process with reason {}", reason);

        if log_enabled!(target: "process", log::Level::Info) {
            firefly_rt::error::printer::print(process).unwrap();
        }

        if process.exception_info.flags.contains(ExceptionFlags::LOG) {
            let mut pid = process.pid();
            let format_args: SmallVec<[OpaqueTerm; 2]> =
                smallvec![Term::Pid(unsafe { Gc::from_raw(&mut pid) }).into(), reason,];
            error_logger::send_error_term_to_logger(
                EXIT_FORMAT,
                format_args,
                process.group_leader().cloned(),
            )
            .ok();
        }

        process.remove_status_flags(StatusFlags::SUSPENDED, Ordering::Relaxed);
        process.set_status_flags(
            StatusFlags::EXITING | StatusFlags::ACTIVE,
            Ordering::Release,
        );
        process.cancel_timer();
        process.reductions += 100;
        process.ip = CONTINUE_EXIT_IP;

        Action::Continue
    }
}

#[derive(Debug)]
#[repr(u8)]
pub enum Action {
    /// This process should continue executing
    Continue,
    /// This process should be rescheduled for later
    Yield,
    /// This process should be suspended indefinitely
    Suspend,
    /// This process was killed/terminated
    Killed,
    /// An error occurred during instruction dispatch
    Error(EmulatorError),
}
impl From<Result<(), EmulatorError>> for Action {
    fn from(result: Result<(), EmulatorError>) -> Self {
        match result {
            Ok(_) => Self::Continue,
            Err(err) => Self::Error(err),
        }
    }
}
impl From<Result<bool, EmulatorError>> for Action {
    fn from(result: Result<bool, EmulatorError>) -> Self {
        match result {
            Ok(true) => Self::Continue,
            Ok(false) => Self::Yield,
            Err(err) => Self::Error(err),
        }
    }
}

trait Inst {
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action;
}

const GC: ops::GarbageCollect = ops::GarbageCollect { fullsweep: false };
const NORMAL_EXIT_IP: usize = 1;
const CONTINUE_EXIT_IP: usize = 2;
const TRAP_IP: usize = 4;

impl Inst for Opcode<Atom> {
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if log_enabled!(target: "process", log::Level::Trace) {
            trace!(target: "process", "dispatching {:?}", self);
        }
        match self {
            Self::Nop(_) => Action::Continue,
            Self::Mov(op) => op.dispatch(emulator, process),
            Self::Cmov(op) => op.dispatch(emulator, process),
            Self::Ret(op) => op.dispatch(emulator, process),
            Self::Br(op) => op.dispatch(emulator, process),
            Self::Brz(op) => op.dispatch(emulator, process),
            Self::Brnz(op) => op.dispatch(emulator, process),
            Self::JumpTable(op) => op.dispatch(emulator, process),
            Self::JumpTableEntry(op) => op.dispatch(emulator, process),
            Self::Call(op) => op.dispatch(emulator, process),
            Self::CallApply2(op) => op.dispatch(emulator, process),
            Self::CallApply3(op) => op.dispatch(emulator, process),
            Self::CallNative(op) => op.dispatch(emulator, process),
            Self::CallStatic(op) => op.dispatch(emulator, process),
            Self::CallIndirect(op) => op.dispatch(emulator, process),
            Self::Enter(op) => op.dispatch(emulator, process),
            Self::EnterApply2(op) => op.dispatch(emulator, process),
            Self::EnterApply3(op) => op.dispatch(emulator, process),
            Self::EnterNative(op) => op.dispatch(emulator, process),
            Self::EnterStatic(op) => op.dispatch(emulator, process),
            Self::EnterIndirect(op) => op.dispatch(emulator, process),
            Self::IsAtom(op) => op.dispatch(emulator, process),
            Self::IsBool(op) => op.dispatch(emulator, process),
            Self::IsNil(op) => op.dispatch(emulator, process),
            Self::IsTuple(op) => op.dispatch(emulator, process),
            Self::IsTupleFetchArity(op) => op.dispatch(emulator, process),
            Self::IsMap(op) => op.dispatch(emulator, process),
            Self::IsCons(op) => op.dispatch(emulator, process),
            Self::IsList(op) => op.dispatch(emulator, process),
            Self::IsInt(op) => op.dispatch(emulator, process),
            Self::IsFloat(op) => op.dispatch(emulator, process),
            Self::IsNumber(op) => op.dispatch(emulator, process),
            Self::IsPid(op) => op.dispatch(emulator, process),
            Self::IsRef(op) => op.dispatch(emulator, process),
            Self::IsPort(op) => op.dispatch(emulator, process),
            Self::IsBinary(op) => op.dispatch(emulator, process),
            Self::IsFunction(op) => op.dispatch(emulator, process),
            Self::LoadNil(op) => op.dispatch(emulator, process),
            Self::LoadBool(op) => op.dispatch(emulator, process),
            Self::LoadAtom(op) => op.dispatch(emulator, process),
            Self::LoadInt(op) => op.dispatch(emulator, process),
            Self::LoadBig(op) => op.dispatch(emulator, process),
            Self::LoadFloat(op) => op.dispatch(emulator, process),
            Self::LoadBinary(op) => op.dispatch(emulator, process),
            Self::LoadBitstring(op) => op.dispatch(emulator, process),
            Self::Not(op) => op.dispatch(emulator, process),
            Self::And(op) => op.dispatch(emulator, process),
            Self::AndAlso(op) => op.dispatch(emulator, process),
            Self::Or(op) => op.dispatch(emulator, process),
            Self::OrElse(op) => op.dispatch(emulator, process),
            Self::Xor(op) => op.dispatch(emulator, process),
            Self::Bnot(op) => op.dispatch(emulator, process),
            Self::Band(op) => op.dispatch(emulator, process),
            Self::Bor(op) => op.dispatch(emulator, process),
            Self::Bxor(op) => op.dispatch(emulator, process),
            Self::Bsl(op) => op.dispatch(emulator, process),
            Self::Bsr(op) => op.dispatch(emulator, process),
            Self::Div(op) => op.dispatch(emulator, process),
            Self::Rem(op) => op.dispatch(emulator, process),
            Self::Neg(op) => op.dispatch(emulator, process),
            Self::Add(op) => op.dispatch(emulator, process),
            Self::Sub(op) => op.dispatch(emulator, process),
            Self::Mul(op) => op.dispatch(emulator, process),
            Self::Divide(op) => op.dispatch(emulator, process),
            Self::ListAppend(op) => op.dispatch(emulator, process),
            Self::ListRemove(op) => op.dispatch(emulator, process),
            Self::Eq(op) => op.dispatch(emulator, process),
            Self::Neq(op) => op.dispatch(emulator, process),
            Self::Gt(op) => op.dispatch(emulator, process),
            Self::Gte(op) => op.dispatch(emulator, process),
            Self::Lt(op) => op.dispatch(emulator, process),
            Self::Lte(op) => op.dispatch(emulator, process),
            Self::Cons(op) => op.dispatch(emulator, process),
            Self::Split(op) => op.dispatch(emulator, process),
            Self::Head(op) => op.dispatch(emulator, process),
            Self::Tail(op) => op.dispatch(emulator, process),
            Self::Closure(op) => op.dispatch(emulator, process),
            Self::UnpackEnv(op) => op.dispatch(emulator, process),
            Self::Tuple(op) => op.dispatch(emulator, process),
            Self::TupleWithCapacity(op) => op.dispatch(emulator, process),
            Self::TupleArity(op) => op.dispatch(emulator, process),
            Self::GetElement(op) => op.dispatch(emulator, process),
            Self::SetElement(op) => op.dispatch(emulator, process),
            Self::SetElementMut(op) => op.dispatch(emulator, process),
            Self::Map(op) => op.dispatch(emulator, process),
            Self::MapPut(op) => op.dispatch(emulator, process),
            Self::MapPutMut(op) => op.dispatch(emulator, process),
            Self::MapUpdate(op) => op.dispatch(emulator, process),
            Self::MapUpdateMut(op) => op.dispatch(emulator, process),
            Self::MapExtendPut(op) => op.dispatch(emulator, process),
            Self::MapExtendUpdate(op) => op.dispatch(emulator, process),
            Self::MapTryGet(op) => op.dispatch(emulator, process),
            Self::Catch(op) => op.dispatch(emulator, process),
            Self::EndCatch(op) => op.dispatch(emulator, process),
            Self::LandingPad(op) => op.dispatch(emulator, process),
            Self::StackTrace(op) => op.dispatch(emulator, process),
            Self::Raise(op) => op.dispatch(emulator, process),
            Self::Send(op) => op.dispatch(emulator, process),
            Self::RecvPeek(op) => op.dispatch(emulator, process),
            Self::RecvNext(op) => op.dispatch(emulator, process),
            Self::RecvWait(op) => op.dispatch(emulator, process),
            Self::RecvTimeout(op) => op.dispatch(emulator, process),
            Self::RecvPop(op) => op.dispatch(emulator, process),
            Self::Await(op) => op.dispatch(emulator, process),
            Self::Yield(op) => op.dispatch(emulator, process),
            Self::GarbageCollect(op) => op.dispatch(emulator, process),
            Self::NormalExit(op) => op.dispatch(emulator, process),
            Self::ContinueExit(op) => op.dispatch(emulator, process),
            Self::Exit1(op) => op.dispatch(emulator, process),
            Self::Exit2(op) => op.dispatch(emulator, process),
            Self::Error1(op) => op.dispatch(emulator, process),
            Self::Throw1(op) => op.dispatch(emulator, process),
            Self::Halt(op) => op.dispatch(emulator, process),
            Self::BsInit(op) => op.dispatch(emulator, process),
            Self::BsPush(op) => op.dispatch(emulator, process),
            Self::BsFinish(op) => op.dispatch(emulator, process),
            Self::BsMatchStart(op) => op.dispatch(emulator, process),
            Self::BsMatch(op) => op.dispatch(emulator, process),
            Self::BsMatchSkip(op) => op.dispatch(emulator, process),
            Self::BsTestTail(op) => op.dispatch(emulator, process),
            Self::FuncInfo(op) => op.dispatch(emulator, process),
            Self::Identity(op) => op.dispatch(emulator, process),
            Self::Spawn2(op) => op.dispatch(emulator, process),
            Self::Spawn3(op) => op.dispatch(emulator, process),
            Self::Spawn3Indirect(op) => op.dispatch(emulator, process),
            Self::Trap(op) => op.dispatch(emulator, process),
        }
    }
}

impl Inst for ops::Ret {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        trace!(target: "process", "returning {}", process.stack.load(self.reg));
        process.stack.copy(self.reg, RETURN_REG);
        let ip = process.stack.pop_frame().unwrap_or(NORMAL_EXIT_IP);
        process.ip = ip;
        Action::Continue
    }
}
impl Inst for ops::Mov {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.copy(self.src, self.dest);
        Action::Continue
    }
}
impl Inst for ops::Cmov {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.cond) {
            OpaqueTerm::TRUE => {
                process.stack.copy(self.src, self.dest);
                Action::Continue
            }
            OpaqueTerm::FALSE => Action::Continue,
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::Br {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.ip = process
            .ip
            .checked_add_signed(self.offset as isize - 1)
            .unwrap();
        Action::Continue
    }
}
impl Inst for ops::Brz {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.reg) {
            OpaqueTerm::FALSE => {
                process.ip = process
                    .ip
                    .checked_add_signed(self.offset as isize - 1)
                    .unwrap();
                Action::Continue
            }
            OpaqueTerm::TRUE => Action::Continue,
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::Brnz {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.reg) {
            OpaqueTerm::TRUE => {
                process.ip = process
                    .ip
                    .checked_add_signed(self.offset as isize - 1)
                    .unwrap();
                Action::Continue
            }
            OpaqueTerm::FALSE => Action::Continue,
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::JumpTable {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.reg).into() {
            Term::Int(value) => {
                let val = value as u32;

                let len = self.len as usize;
                let arms = &emulator.code.code[process.ip..(process.ip + len)];
                for (i, arm) in arms.iter().enumerate() {
                    let Opcode::JumpTableEntry(ops::JumpTableEntry { imm, offset }) = arm else { unreachable!() };
                    if val.eq(imm) {
                        process.ip = process
                            .ip
                            .checked_add_signed(*offset as isize + i as isize)
                            .unwrap();
                        return Action::Continue;
                    }
                }

                // If we reach here, none of the arms matched, jump to next instruction
                // after the last jump table entry. We have to subtract one from the table
                // length because on entry the instruction pointer is already pointing to
                // the first entry.
                process.ip += len - 1;
                Action::Continue
            }
            other => panic!("expected integer immediate, but got `{}`", &other),
        }
    }
}
/// This instruction should never be dispatched, but if it is, treat it as an unconditional branch
impl Inst for ops::JumpTableEntry {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.ip = process
            .ip
            .checked_add_signed(self.offset as isize - 1)
            .unwrap();
        Action::Continue
    }
}
impl Inst for ops::Call {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // When a Call is performed, `dest` is at the bottom of the registers window,
        // followed by an uninitialized register to store the return address, then all
        // of the callee arguments:
        //
        //     0  | dest       <- fp
        //     1  | process.ip
        //     2  | arg1
        //     .
        //     .  |
        //     N  | argN
        //     N+1| NONE       <- sp
        //
        // We must start a new call frame and write the return address before transferring
        // control to the callee.
        process.stack.push_frame(self.dest);
        let cp = OpaqueTerm::code(process.ip);
        process.stack.store(CP_REG, cp);
        process.ip = self.offset as usize;
        Action::Continue
    }
}
impl Inst for ops::CallApply2 {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Move the argument list to the stack
        let arglist = process.stack.load(self.argv);
        let mut arity = 0;
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for (i, result) in cons.iter_raw().enumerate() {
                    if result.is_err() {
                        unsafe {
                            process.stack.dealloc(arity as usize);
                        }
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    arity += 1;
                    unsafe {
                        process.stack.alloca(1);
                    }
                    let reg = self.dest + 2 + (i as Register);
                    process
                        .stack
                        .store(reg, unsafe { result.unwrap_unchecked() });
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }
        // Convert to a CallIndirect
        let op = ops::CallIndirect {
            dest: self.dest,
            callee: self.callee,
            arity: arity as u8,
        };
        op.dispatch(emulator, process)
    }
}
impl Inst for ops::CallApply3 {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Move the argument list to the stack
        let arglist = process.stack.load(self.argv);
        let mut arity = 0;
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for (i, result) in cons.iter_raw().enumerate() {
                    if result.is_err() {
                        unsafe {
                            process.stack.dealloc(arity);
                        }
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    arity += 1;
                    unsafe {
                        process.stack.alloca(1);
                    }
                    let reg = self.dest + 2 + (i as Register);
                    process
                        .stack
                        .store(reg, unsafe { result.unwrap_unchecked() });
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }

        let module = process.stack.load(self.module);
        let function = process.stack.load(self.function);
        match (module.into(), function.into()) {
            (Term::Atom(m), Term::Atom(f)) => {
                let mfa = bc::ModuleFunctionArity {
                    module: m,
                    function: f,
                    arity: arity as u8,
                };
                match emulator.code.function_by_mfa(&mfa).map(|fun| fun.id()) {
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    Some(id) => {
                        // Convert to a CallStatic
                        let op = ops::CallStatic {
                            dest: self.dest,
                            callee: id,
                        };
                        op.dispatch(emulator, process)
                    }
                }
            }
            (Term::Atom(_), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = function;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = module;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }
    }
}
impl Inst for ops::CallNative {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let callee = unsafe { mem::transmute::<_, DynamicCallee>(self.callee) };
        process.stack.push_frame(self.dest);
        let cp = OpaqueTerm::code(process.ip);
        process.stack.store(CP_REG, cp);
        let argv = process
            .stack
            .select_registers(ARG0_REG, self.arity as usize);
        let argc = argv.len();
        let argv = argv.as_ptr();
        match unsafe { function::dynamic::apply(callee, process, argv, argc) } {
            ErlangResult::Ok(result) => {
                process.stack.store(RETURN_REG, result);
                let op = ops::Ret { reg: RETURN_REG };
                op.dispatch(emulator, process)
            }
            ErlangResult::Err | ErlangResult::Exit => emulator.handle_error(process),
            ErlangResult::Await(generator) => {
                // We're blocked on some generator which needs to be run to completion
                process.awaiting = Some(Box::into_inner(generator));
                Action::Yield
            }
            ErlangResult::Trap(mfa) => {
                // The native function is trapping, so we are playing the
                // role of a trampoline here and tail calling the trap function.
                //
                // The trapping NIF/BIF will have already placed the callee arguments
                // on the stack in their appropriate argument slots, so we need only
                // dispatch the call itself. However we must also handle the case where
                // the desired callee doesn't exist, however unlikely that may be.
                //
                // To ensure we yield to the scheduler if we're out of reductions, we
                // dispatch to the callee via the `Trap` instruction, but first we must
                // store the callee function id in the process state.
                let mfa = (*mfa).into();
                match emulator.code.function_by_mfa(&mfa).map(|fun| fun.id()) {
                    Some(callee) => {
                        process.trap = Some(callee);
                        process.ip = TRAP_IP;
                        Action::Continue
                    }
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
        }
    }
}
impl Inst for ops::CallStatic {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // When a Call is performed, `dest` is at the bottom of the registers window,
        // followed by an uninitialized register to store the return address, then all
        // of the callee arguments:
        //
        //     0  | dest       <- fp
        //     1  | process.ip
        //     2  | arg1
        //     .
        //     .  |
        //     N  | argN
        //     N+1| NONE       <- sp
        //
        // We must start a new call frame and write the return address before transferring
        // control to the callee.
        match emulator.code.function_by_id(self.callee) {
            Function::Bytecode {
                offset,
                is_nif: false,
                ..
            } => {
                let op = ops::Call {
                    dest: self.dest,
                    offset: *offset,
                };
                op.dispatch(emulator, process)
            }
            Function::Bytecode { mfa, offset, .. } => {
                let mfa = (*mfa).into();
                // Try to call the native implementation
                match function::find_symbol(&mfa) {
                    Some(symbol) => {
                        let op = ops::CallNative {
                            dest: self.dest,
                            arity: mfa.arity,
                            callee: symbol as *const (),
                        };
                        op.dispatch(emulator, process)
                    }
                    _ => {
                        // Fall back to the bytecode definition
                        let op = ops::Call {
                            dest: self.dest,
                            offset: *offset,
                        };
                        op.dispatch(emulator, process)
                    }
                }
            }
            Function::Native { name, arity, .. } => {
                match function::find_native_symbol::<DynamicCallee>(name.as_str().as_bytes()) {
                    Ok(symbol) => {
                        let op = ops::CallNative {
                            dest: self.dest,
                            arity: *arity,
                            callee: unsafe {
                                mem::transmute::<DynamicCallee, *const ()>(*symbol.deref())
                            },
                        };
                        op.dispatch(emulator, process)
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
            Function::Bif { mfa, .. } => {
                let mfa = (*mfa).into();
                match function::find_symbol(&mfa) {
                    Some(symbol) => {
                        let op = ops::CallNative {
                            dest: self.dest,
                            arity: mfa.arity,
                            callee: symbol as *const (),
                        };
                        op.dispatch(emulator, process)
                    }
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
        }
    }
}
impl Inst for ops::Enter {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // An `Enter` is a tail call, reusing the callers frame.
        process.ip = self.offset as usize;
        Action::Continue
    }
}
impl Inst for ops::EnterApply2 {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Since the argument list contains all of the arguments, we don't need
        // to worry about clobbering any of the registers once we load the callee
        let callee = process.stack.load(self.callee);
        let arglist = process.stack.load(self.argv);
        // Capture the size of this frame in case our callee has more arguments than
        // the current frame has available slots
        let mut available = process.stack.stack_pointer() - 2 - process.stack.frame_pointer();
        // Arity is variable depending on whether the closure is thin or not
        let mut arity = 0;
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for (i, result) in cons.iter_raw().enumerate() {
                    if result.is_err() {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    arity += 1;
                    match available.checked_sub(1) {
                        Some(new_available) => {
                            available = new_available;
                            process.stack.store(ARG0_REG + i as Register, unsafe {
                                result.unwrap_unchecked()
                            });
                        }
                        None => {
                            // We've used up all the stack slots that were previously allocated,
                            // we need to allocate more dynamically from this point on.
                            unsafe {
                                process.stack.alloca(1);
                            }
                            process.stack.store(ARG0_REG + i as Register, unsafe {
                                result.unwrap_unchecked()
                            });
                        }
                    }
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }
        // Convert to EnterIndirect
        let callee = match available.checked_sub(1) {
            Some(_) => {
                let reg = ARG0_REG + arity as Register;
                process.stack.store(reg, callee);
                reg
            }
            None => {
                unsafe {
                    process.stack.alloca(1);
                }
                let reg = ARG0_REG + arity as Register;
                process.stack.store(reg, callee);
                reg
            }
        };
        let op = ops::EnterIndirect {
            callee,
            arity: arity as u8,
        };
        op.dispatch(emulator, process)
    }
}
impl Inst for ops::EnterApply3 {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Since the argument list contains all of the arguments, we don't need
        // to worry about clobbering any of the registers once we load the module/function
        let module = process.stack.load(self.module);
        let function = process.stack.load(self.function);
        let arglist = process.stack.load(self.argv);
        // Capture the size of this frame in case our callee has more arguments than
        // the current frame has available slots
        let mut available = process.stack.stack_pointer() - 2 - process.stack.frame_pointer();
        let mut arity = 0;
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for (i, result) in cons.iter_raw().enumerate() {
                    if result.is_err() {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    arity += 1;
                    match available.checked_sub(1) {
                        Some(new_available) => {
                            available = new_available;
                            process.stack.store(ARG0_REG + i as Register, unsafe {
                                result.unwrap_unchecked()
                            });
                        }
                        None => {
                            // We've used up all the stack slots that were previously allocated,
                            // we need to allocate more dynamically from this point on.
                            unsafe {
                                process.stack.alloca(1);
                            }
                            process.stack.store(ARG0_REG + i as Register, unsafe {
                                result.unwrap_unchecked()
                            });
                        }
                    }
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }

        match (module.into(), function.into()) {
            (Term::Atom(m), Term::Atom(f)) => {
                let mfa = bc::ModuleFunctionArity {
                    module: m,
                    function: f,
                    arity: arity as u8,
                };
                match emulator.code.function_by_mfa(&mfa).map(|fun| fun.id()) {
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    Some(id) => {
                        // Convert to a EnterStatic
                        let op = ops::EnterStatic { callee: id };
                        op.dispatch(emulator, process)
                    }
                }
            }
            (Term::Atom(_), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = function;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = module;
                process.exception_info.trace = None;
                return emulator.handle_error(process);
            }
        }
    }
}
impl Inst for ops::EnterNative {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let callee = unsafe { mem::transmute::<_, DynamicCallee>(self.callee) };
        let argv = process
            .stack
            .select_registers(ARG0_REG, self.arity as usize);
        let argc = argv.len();
        let argv = argv.as_ptr();
        match unsafe { function::dynamic::apply(callee, process, argv, argc) } {
            ErlangResult::Ok(result) => {
                process.stack.store(RETURN_REG, result);
                let op = ops::Ret { reg: RETURN_REG };
                op.dispatch(emulator, process)
            }
            ErlangResult::Err | ErlangResult::Exit => emulator.handle_error(process),
            ErlangResult::Await(generator) => {
                // See the comment in CallNative regarding awaits
                process.awaiting = Some(Box::into_inner(generator));
                Action::Yield
            }
            ErlangResult::Trap(mfa) => {
                // See the comment in CallNative regarding traps
                let mfa = (*mfa).into();
                match emulator.code.function_by_mfa(&mfa).map(|fun| fun.id()) {
                    Some(callee) => {
                        process.trap = Some(callee);
                        process.ip = TRAP_IP;
                        Action::Continue
                    }
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
        }
    }
}
impl Inst for ops::EnterStatic {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // This is similar to a call, but represents a tail call, where we are reusing the
        // caller's frame. At the point where we encounter this instruction, everything has
        // already been prepared, so we can immediately transfer control to the callee.
        match emulator.code.function_by_id(self.callee) {
            Function::Bytecode {
                offset,
                is_nif: false,
                ..
            } => {
                let op = ops::Enter { offset: *offset };
                op.dispatch(emulator, process)
            }
            Function::Bytecode { mfa, offset, .. } => {
                let mfa = (*mfa).into();
                match function::find_symbol(&mfa) {
                    Some(symbol) => {
                        let op = ops::EnterNative {
                            callee: symbol as *const (),
                            arity: mfa.arity,
                        };
                        op.dispatch(emulator, process)
                    }
                    _ => {
                        // Fall back to the bytecode definition
                        let op = ops::Enter { offset: *offset };
                        op.dispatch(emulator, process)
                    }
                }
            }
            Function::Native { name, arity, .. } => {
                match function::find_native_symbol::<DynamicCallee>(name.as_str().as_bytes()) {
                    Ok(symbol) => {
                        let op = ops::EnterNative {
                            callee: unsafe {
                                mem::transmute::<DynamicCallee, *const ()>(*symbol.deref())
                            },
                            arity: *arity,
                        };
                        op.dispatch(emulator, process)
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
            Function::Bif { mfa, .. } => {
                let mfa = (*mfa).into();
                match function::find_symbol(&mfa) {
                    Some(symbol) => {
                        let op = ops::EnterNative {
                            callee: symbol as *const (),
                            arity: mfa.arity,
                        };
                        op.dispatch(emulator, process)
                    }
                    None => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Undef.into();
                        process.exception_info.value = atoms::Undef.into();
                        process.exception_info.trace = None;
                        emulator.handle_error(process)
                    }
                }
            }
        }
    }
}
impl Inst for ops::CallIndirect {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Same as Call, but the callee is a Closure term in a register
        let callee = process.stack.load(self.callee);
        match callee.into() {
            Term::Closure(fun) => {
                let is_thin = fun.is_thin();
                let expected_arity = (!is_thin as u8) + self.arity;
                if fun.arity == expected_arity {
                    // If the callee is a proper closure, we need to introduce the closure argument
                    // at the end
                    if !is_thin {
                        process
                            .stack
                            .store(self.dest + 2 + self.arity as Register, callee);
                    }
                    if fun.is_native() {
                        let op = ops::CallNative {
                            dest: self.dest,
                            callee: fun.callee,
                            arity: expected_arity,
                        };
                        op.dispatch(emulator, process)
                    } else {
                        let op = ops::Call {
                            dest: self.dest,
                            offset: fun.callee as usize,
                        };
                        op.dispatch(emulator, process)
                    }
                } else {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Undef.into();
                    process.exception_info.value = atoms::Undef.into();
                    process.exception_info.trace = None;
                    emulator.handle_error(process)
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badfun.into();
                process.exception_info.value = atoms::Badfun.into();
                process.exception_info.trace = None;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::EnterIndirect {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // Same as enter, but the callee is a Closure term in a register
        let callee = process.stack.load(self.callee);
        match callee.into() {
            Term::Closure(fun) => {
                let is_thin = fun.is_thin();
                let expected_arity = (!is_thin as u8) + self.arity;
                if fun.arity == expected_arity {
                    if !is_thin {
                        process
                            .stack
                            .store(ARG0_REG + self.arity as Register, callee);
                    }
                    if fun.is_native() {
                        let op = ops::EnterNative {
                            callee: fun.callee,
                            arity: expected_arity,
                        };
                        op.dispatch(emulator, process)
                    } else {
                        let op = ops::Enter {
                            offset: fun.callee as usize,
                        };
                        op.dispatch(emulator, process)
                    }
                } else {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Undef.into();
                    process.exception_info.value = atoms::Undef.into();
                    process.exception_info.trace = None;
                    emulator.handle_error(process)
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badfun.into();
                process.exception_info.value = atoms::Badfun.into();
                process.exception_info.trace = None;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::IsAtom {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_atom = process.stack.load(self.value).is_atom();
        process.stack.store(self.dest, is_atom.into());
        Action::Continue
    }
}
impl Inst for ops::IsBool {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.value) {
            OpaqueTerm::TRUE | OpaqueTerm::FALSE => {
                process.stack.store(self.dest, OpaqueTerm::TRUE)
            }
            _ => process.stack.store(self.dest, OpaqueTerm::FALSE),
        }
        Action::Continue
    }
}
impl Inst for ops::IsNil {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_nil = process.stack.load(self.value).is_nil();
        process.stack.store(self.dest, is_nil.into());
        Action::Continue
    }
}
impl Inst for ops::IsTuple {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_tuple = process.stack.load(self.value).is_tuple();
        process.stack.store(self.dest, is_tuple.into());
        Action::Continue
    }
}
impl Inst for ops::IsTupleFetchArity {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let term = process.stack.load(self.value);
        match term.tuple_size() {
            Ok(arity) => {
                process.stack.store(self.dest, OpaqueTerm::TRUE);
                process
                    .stack
                    .store(self.arity, Term::Int(arity as i64).into());
            }
            Err(_) => {
                process.stack.store(self.dest, OpaqueTerm::FALSE);
                process.stack.store(self.arity, OpaqueTerm::NONE);
            }
        }
        Action::Continue
    }
}
impl Inst for ops::IsMap {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_map = match process.stack.load(self.value).r#typeof() {
            TermType::Map => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_map);
        Action::Continue
    }
}
impl Inst for ops::IsCons {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_cons = process.stack.load(self.value).is_nonempty_list();
        process.stack.store(self.dest, is_cons.into());
        Action::Continue
    }
}
impl Inst for ops::IsList {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_list = process.stack.load(self.value).is_list();
        process.stack.store(self.dest, is_list.into());
        Action::Continue
    }
}
impl Inst for ops::IsInt {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_integer = match process.stack.load(self.value).r#typeof() {
            TermType::Int => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_integer);
        Action::Continue
    }
}
impl Inst for ops::IsFloat {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_float = process.stack.load(self.value).is_float();
        process.stack.store(self.dest, is_float.into());
        Action::Continue
    }
}
impl Inst for ops::IsNumber {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_number = process.stack.load(self.value).r#typeof().is_number();
        process.stack.store(self.dest, is_number.into());
        Action::Continue
    }
}
impl Inst for ops::IsPid {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_pid = match process.stack.load(self.value).r#typeof() {
            TermType::Pid => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_pid);
        Action::Continue
    }
}
impl Inst for ops::IsPort {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_port = match process.stack.load(self.value).r#typeof() {
            TermType::Port => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_port);
        Action::Continue
    }
}
impl Inst for ops::IsRef {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_ref = match process.stack.load(self.value).r#typeof() {
            TermType::Reference => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_ref);
        Action::Continue
    }
}
impl Inst for ops::IsBinary {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let is_binary = match process.stack.load(self.value).r#typeof() {
            TermType::Binary => OpaqueTerm::TRUE,
            _ => OpaqueTerm::FALSE,
        };
        process.stack.store(self.dest, is_binary);
        Action::Continue
    }
}
impl Inst for ops::IsFunction {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let fun_term = process.stack.load(self.value);
        match fun_term.into() {
            Term::Closure(_) if self.arity.is_none() => {
                process.stack.store(self.dest, true.into());
            }
            Term::Closure(fun) => {
                let arity_term = process.stack.load(self.arity.unwrap());
                match arity_term.into() {
                    Term::Int(i) if i >= 0 => {
                        process
                            .stack
                            .store(self.dest, (fun.arity as i64 == i).into());
                    }
                    _ => {
                        process.exception_info = ExceptionInfo::error(atoms::Badarg.into());
                        process.exception_info.value = arity_term;
                        return emulator.handle_error(process);
                    }
                }
            }
            _ => {
                process.stack.store(self.dest, false.into());
            }
        }
        Action::Continue
    }
}
impl Inst for ops::LoadNil {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.store(self.dest, OpaqueTerm::NIL);
        Action::Continue
    }
}
impl Inst for ops::LoadBool {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.store(self.dest, self.value.into());
        Action::Continue
    }
}
impl Inst for ops::LoadAtom<Atom> {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.store(self.dest, self.value.into());
        Action::Continue
    }
}
impl Inst for ops::LoadInt {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if let Ok(term) = self.value.try_into() {
            process.stack.store(self.dest, term);
            Action::Continue
        } else {
            if let Ok(mut empty) = Gc::new_uninit_in(process) {
                let term = unsafe {
                    empty.write(BigInt::from(self.value));
                    empty.assume_init()
                };
                process.stack.store(self.dest, term.into());
                Action::Continue
            } else {
                process.gc_needed = mem::size_of::<BigInt>();
                process.ip -= 1;
                GC.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::LoadBig {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if let Ok(mut empty) = Gc::new_uninit_in(&process) {
            let term = unsafe {
                empty.write(BigInt::from(self.value.clone()));
                Term::BigInt(empty.assume_init())
            };
            process.stack.store(self.dest, term.into());
            Action::Continue
        } else {
            // Dispatch to the garbage collect op, but reset the instruction pointer to this op
            // When the gc finishes executing, it will resume execution at this op again, but this
            // time with sufficient space to succeed
            process.gc_needed = mem::size_of::<BigInt>();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::LoadFloat {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.store(self.dest, self.value.into());
        Action::Continue
    }
}
impl Inst for ops::LoadBinary {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let data: &'static BinaryData = unsafe { &*(self.value as *const BinaryData) };
        process.stack.store(self.dest, data.into());
        Action::Continue
    }
}
impl Inst for ops::LoadBitstring {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let data: &'static BinaryData = unsafe { &*(self.value as *const BinaryData) };
        process.stack.store(self.dest, data.into());
        Action::Continue
    }
}
impl Inst for ops::Cons {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if let Ok(mut cell) = Cons::new_uninit_in(&process) {
            let cell = unsafe {
                cell.write(Cons {
                    head: process.stack.load(self.head),
                    tail: process.stack.load(self.tail),
                });
                cell.assume_init()
            };
            process.stack.store(self.dest, cell.into());
            Action::Continue
        } else {
            // Must perform a GC before we can proceed
            process.gc_needed = mem::size_of::<Cons>();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::Split {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.list).into() {
            Term::Cons(cons) => {
                process.stack.store(self.hd, cons.head);
                process.stack.store(self.tl, cons.tail);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::Head {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let term = process.stack.load(self.list);
        match term.into() {
            Term::Cons(cons) => {
                process.stack.store(self.dest, cons.head);
                Action::Continue
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = term;
                process.exception_info.trace = None;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::Tail {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let term = process.stack.load(self.list);
        match term.into() {
            Term::Cons(cons) => {
                process.stack.store(self.dest, cons.tail);
                Action::Continue
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = term;
                process.exception_info.trace = None;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::Closure {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let env = process
            .stack
            .select_registers(self.dest + 1, self.arity as usize);
        let f = emulator.code.function_by_id(self.function);
        let module;
        let function;
        let arity;
        let mut flags = ClosureFlags::empty();
        let callee;
        match f {
            Function::Bytecode {
                offset,
                mfa,
                is_nif,
                ..
            } => {
                let mfa: ModuleFunctionArity = (*mfa).into();
                module = mfa.module;
                function = mfa.function;
                arity = mfa.arity;
                let offset = *offset;
                if *is_nif {
                    match function::find_symbol(&mfa) {
                        None => {
                            flags = ClosureFlags::BYTECODE;
                            callee = offset as *const ();
                        }
                        Some(ptr) => {
                            callee = ptr as *const ();
                        }
                    }
                } else {
                    flags = ClosureFlags::BYTECODE;
                    callee = offset as *const ();
                }
            }
            Function::Bif { mfa, .. } => {
                let mfa: ModuleFunctionArity = (*mfa).into();
                module = mfa.module;
                function = mfa.function;
                arity = mfa.arity;
                if let Some(ptr) = function::find_symbol(&mfa) {
                    callee = ptr as *const ();
                } else {
                    todo!("undefined function")
                }
            }
            Function::Native { name, arity: a, .. } => {
                module = atoms::Undefined;
                function = *name;
                arity = *a;
                match function::find_native_symbol::<DynamicCallee>(name.as_str().as_bytes()) {
                    Ok(symbol) => {
                        callee =
                            unsafe { mem::transmute::<DynamicCallee, *const ()>(*symbol.deref()) };
                    }
                    Err(_) => todo!("undefined function"),
                }
            }
        }
        if let Ok(closure) =
            Closure::new_with_flags_in(module, function, arity, flags, callee, env, process)
        {
            process.stack.store(self.dest, closure.into());
            Action::Continue
        } else {
            let mut builder = LayoutBuilder::new();
            builder.build_closure(env.len());
            process.gc_needed = builder.finish().size();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::UnpackEnv {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.fun).into() {
            Term::Closure(fun) => {
                let value = fun.env()[self.index as usize];
                process.stack.store(self.dest, value);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::Tuple {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let elems = process
            .stack
            .select_registers(self.dest + 1, self.arity as usize);
        if let Ok(tuple) = Tuple::from_slice(elems, process) {
            process.stack.store(self.dest, tuple.into());
            Action::Continue
        } else {
            let mut builder = LayoutBuilder::new();
            builder.build_tuple(self.arity as usize);
            process.gc_needed = builder.finish().size();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::TupleWithCapacity {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if let Ok(tuple) = Tuple::new_in(self.arity as usize, process) {
            process.stack.store(self.dest, tuple.into());
            Action::Continue
        } else {
            let mut builder = LayoutBuilder::new();
            builder.build_tuple(self.arity as usize);
            process.gc_needed = builder.finish().size();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::TupleArity {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.tuple).tuple_size() {
            Ok(arity) => {
                process
                    .stack
                    .store(self.dest, Term::Int(arity as i64).into());
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::GetElement {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.tuple).into() {
            Term::Tuple(tuple) => {
                process.stack.store(self.dest, tuple[self.index as usize]);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::SetElement {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.tuple).into() {
            Term::Tuple(tuple) => {
                let value = process.stack.load(self.value);
                if let Ok(updated) = tuple.set_element(self.index as usize, value, process) {
                    process.stack.store(self.dest, updated.into());
                    Action::Continue
                } else {
                    process.gc_needed = mem::size_of_val(tuple.deref());
                    process.ip -= 1;
                    GC.dispatch(emulator, process)
                }
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::SetElementMut {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.tuple).into() {
            Term::Tuple(mut tuple) => {
                let value = process.stack.load(self.value);
                tuple.set_element_mut(self.index as usize, value);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::Map {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if let Ok(map) = Map::with_capacity_in(self.capacity, process) {
            process.stack.store(self.dest, map.into());
            Action::Continue
        } else {
            let mut builder = LayoutBuilder::new();
            builder.build_map(self.capacity);
            process.gc_needed = builder.finish().size();
            process.ip -= 1;
            GC.dispatch(emulator, process)
        }
    }
}
impl Inst for ops::MapPut {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.map).into() {
            Term::Map(map) => {
                let key = process.stack.load(self.key);
                let value = process.stack.load(self.value);
                match map.put(key, value, process) {
                    Ok(updated) => {
                        process.stack.store(self.dest, updated.into());
                        Action::Continue
                    }
                    Err(_) => {
                        // Reserve an extra two words in case the insert would extend the map
                        process.gc_needed = mem::size_of_val(map.deref()) + 16;
                        process.ip -= 1;
                        GC.dispatch(emulator, process)
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapPutMut {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.map).into() {
            Term::Map(mut map) => {
                let key = process.stack.load(self.key);
                let value = process.stack.load(self.value);
                map.put_mut(key, value);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapUpdate {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.map).into() {
            Term::Map(map) => {
                let key = process.stack.load(self.key);
                let value = process.stack.load(self.value);
                match map.update(key, value, process) {
                    Ok(updated) => {
                        process.stack.store(self.dest, updated.into());
                        Action::Continue
                    }
                    Err(MapError::BadKey) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::BadKey.into();
                        process.exception_info.value = key.into();
                        emulator.handle_error(process)
                    }
                    Err(MapError::AllocError(_)) => {
                        process.gc_needed = mem::size_of_val(map.deref());
                        process.ip -= 1;
                        GC.dispatch(emulator, process)
                    }
                    ref err => panic!("unexpected map update error {:?}", err),
                }
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapUpdateMut {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.map).into() {
            Term::Map(mut map) => {
                let key = process.stack.load(self.key);
                if map.contains_key(key) {
                    let value = process.stack.load(self.value);
                    map.put_mut(key, value);
                    Action::Continue
                } else {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::BadKey.into();
                    process.exception_info.value = key;
                    emulator.handle_error(process)
                }
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapExtendPut {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        assert_eq!(self.pairs.len() % 2, 0);
        let term = process.stack.load(self.map);
        match term.into() {
            Term::Map(mut map) => {
                let additional = self.pairs.len() / 2;
                let orig_capacity = map.capacity();
                let orig_size = map.size();
                let free = orig_capacity - orig_size;
                if additional <= free {
                    // We have sufficient free capacity
                    let chunks = unsafe { self.pairs.as_chunks_unchecked() };
                    for [k, v] in chunks {
                        let key = process.stack.load(*k);
                        let value = process.stack.load(*v);
                        map.put_mut(key, value);
                    }
                    process.stack.store(self.dest, term);
                    return Action::Continue;
                }
                // We probably don't, so allocate a new map
                match Map::with_capacity_in(orig_size + additional, process) {
                    Ok(mut extended) => {
                        for (k, v) in map.keys().iter().copied().zip(map.values().iter().copied()) {
                            extended.put_mut(k, v);
                        }
                        let chunks = unsafe { self.pairs.as_chunks_unchecked() };
                        for [k, v] in chunks {
                            let key = process.stack.load(*k);
                            let value = process.stack.load(*v);
                            extended.put_mut(key, value);
                        }
                        process.stack.store(self.dest, extended.into());
                        Action::Continue
                    }
                    Err(_) => {
                        // Reserve an extra two words in case the insert would extend the map
                        process.gc_needed = mem::size_of_val(map.deref()) + additional;
                        process.ip -= 1;
                        GC.dispatch(emulator, process)
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapExtendUpdate {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        assert_eq!(self.pairs.len() % 2, 0);
        let term = process.stack.load(self.map);
        match term.into() {
            Term::Map(mut map) => {
                let chunks = unsafe { self.pairs.as_chunks_unchecked() };
                for [k, v] in chunks {
                    let key = process.stack.load(*k);
                    let value = process.stack.load(*v);
                    if map.contains_key(key) {
                        map.put_mut(key, value);
                    } else {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::BadKey.into();
                        process.exception_info.value = key.into();
                        return emulator.handle_error(process);
                    }
                }
                process.stack.store(self.dest, term);
                Action::Continue
            }
            _ => unreachable!(),
        }
    }
}
impl Inst for ops::MapTryGet {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match process.stack.load(self.map).into() {
            Term::Map(map) => {
                let key = process.stack.load(self.key);
                match map.get(key) {
                    None => {
                        process.stack.store(self.is_err, OpaqueTerm::TRUE);
                        process.stack.store(self.value, OpaqueTerm::NONE);
                    }
                    Some(value) => {
                        process.stack.store(self.is_err, OpaqueTerm::FALSE);
                        process.stack.store(self.value, value);
                    }
                }
                Action::Continue
            }
            _ => {
                process.stack.store(self.is_err, OpaqueTerm::TRUE);
                process.stack.store(self.value, OpaqueTerm::NONE);
                Action::Continue
            }
        }
    }
}
impl Inst for ops::Not {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let inverted = match process.stack.load(self.cond) {
            OpaqueTerm::TRUE => OpaqueTerm::FALSE,
            OpaqueTerm::FALSE => OpaqueTerm::TRUE,
            value => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = value;
                return emulator.handle_error(process);
            }
        };
        process.stack.store(self.dest, inverted);
        Action::Continue
    }
}
impl Inst for ops::And {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Bool(l), Term::Bool(r)) => (l && r).into(),
            (Term::Bool(_), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        process.stack.store(self.dest, result);
        Action::Continue
    }
}
impl Inst for ops::AndAlso {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        if let Term::Bool(l) = lhs.into() {
            if l {
                let rhs = process.stack.load(self.rhs);
                process.stack.store(self.dest, rhs);
            } else {
                process.stack.store(self.dest, lhs);
            }
            Action::Continue
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Or {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Bool(l), Term::Bool(r)) => (l || r).into(),
            (Term::Bool(_), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        process.stack.store(self.dest, result);
        Action::Continue
    }
}
impl Inst for ops::OrElse {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        if let Term::Bool(l) = lhs.into() {
            if l {
                process.stack.store(self.dest, lhs);
            } else {
                let rhs = process.stack.load(self.rhs);
                process.stack.store(self.dest, rhs);
            }
            Action::Continue
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Xor {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Bool(l), Term::Bool(r)) => (l ^ r).into(),
            (Term::Bool(_), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        process.stack.store(self.dest, result);
        Action::Continue
    }
}
impl Inst for ops::Bnot {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let rhs = process.stack.load(self.rhs);
        let result = match rhs.into() {
            Term::Int(l) => Int::Small(!l),
            Term::BigInt(l) => Int::Big(!l.inner()),
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Band {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l & r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l & r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l & r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l & r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Bor {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l | r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l | r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l | r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l | r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Bxor {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l ^ r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l ^ r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l ^ r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l ^ r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Bsl {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.value);
        let rhs = process.stack.load(self.shift);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l << r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l << r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l << r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l << r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Bsr {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.value);
        let rhs = process.stack.load(self.shift);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l >> r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l >> r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l >> r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l >> r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Int::Small(i) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Int::Big(i) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Div {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.value);
        let rhs = process.stack.load(self.divisor);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l / r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l / r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l / r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l / r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Ok(Int::Small(i)) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Ok(Int::Big(i)) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
            Err(_) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
        }
    }
}
impl Inst for ops::Rem {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.value);
        let rhs = process.stack.load(self.divisor);
        let result = match (lhs.into(), rhs.into()) {
            (Term::Int(l), Term::Int(r)) => {
                let l = Int::Small(l);
                let r = Int::Small(r);
                l % r
            }
            (Term::Int(l), Term::BigInt(r)) => {
                let l = Int::Small(l);
                let r = Int::Big(r.inner().clone());
                l % r
            }
            (Term::BigInt(l), Term::Int(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Small(r);
                l % r
            }
            (Term::BigInt(l), Term::BigInt(r)) => {
                let l = Int::Big(l.inner().clone());
                let r = Int::Big(r.inner().clone());
                l % r
            }
            (_int @ (Term::Int(_) | Term::BigInt(_)), _other) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = lhs;
                return emulator.handle_error(process);
            }
        };
        match result {
            Ok(Int::Small(i)) => {
                process.stack.store(self.dest, Term::Int(i).into());
                Action::Continue
            }
            Ok(Int::Big(i)) => {
                let op = ops::LoadBig {
                    dest: self.dest,
                    value: i,
                };
                op.dispatch(emulator, process)
            }
            Err(_) => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                return emulator.handle_error(process);
            }
        }
    }
}
impl Inst for ops::Neg {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let rhs = process.stack.load(self.rhs);
        let term: Term = rhs.into();
        if let Ok(num) = TryInto::<Number>::try_into(term) {
            match -num {
                Number::Integer(Int::Small(i)) => {
                    process.stack.store(self.dest, Term::Int(i).into());
                    Action::Continue
                }
                Number::Integer(Int::Big(i)) => {
                    let op = ops::LoadBig {
                        dest: self.dest,
                        value: i,
                    };
                    op.dispatch(emulator, process)
                }
                Number::Float(f) => {
                    process.stack.store(self.dest, Term::Float(f).into());
                    Action::Continue
                }
            }
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = rhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Add {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let lterm: Term = lhs.into();
        if let Ok(l) = TryInto::<Number>::try_into(lterm) {
            let rhs = process.stack.load(self.rhs);
            let rterm: Term = rhs.into();
            if let Ok(r) = TryInto::<Number>::try_into(rterm) {
                match l + r {
                    Ok(Number::Integer(Int::Small(i))) => {
                        process.stack.store(self.dest, Term::Int(i).into());
                        Action::Continue
                    }
                    Ok(Number::Integer(Int::Big(i))) => {
                        let op = ops::LoadBig {
                            dest: self.dest,
                            value: i,
                        };
                        op.dispatch(emulator, process)
                    }
                    Ok(Number::Float(f)) => {
                        process.stack.store(self.dest, Term::Float(f).into());
                        Action::Continue
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = rhs;
                        emulator.handle_error(process)
                    }
                }
            } else {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                emulator.handle_error(process)
            }
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Sub {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let lterm: Term = lhs.into();
        if let Ok(l) = TryInto::<Number>::try_into(lterm) {
            let rhs = process.stack.load(self.rhs);
            let rterm: Term = rhs.into();
            if let Ok(r) = TryInto::<Number>::try_into(rterm) {
                match l - r {
                    Ok(Number::Integer(Int::Small(i))) => {
                        process.stack.store(self.dest, Term::Int(i).into());
                        Action::Continue
                    }
                    Ok(Number::Integer(Int::Big(i))) => {
                        let op = ops::LoadBig {
                            dest: self.dest,
                            value: i,
                        };
                        op.dispatch(emulator, process)
                    }
                    Ok(Number::Float(f)) => {
                        process.stack.store(self.dest, Term::Float(f).into());
                        Action::Continue
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = rhs;
                        emulator.handle_error(process)
                    }
                }
            } else {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                emulator.handle_error(process)
            }
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Mul {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let lterm: Term = lhs.into();
        if let Ok(l) = TryInto::<Number>::try_into(lterm) {
            let rhs = process.stack.load(self.rhs);
            let rterm: Term = rhs.into();
            if let Ok(r) = TryInto::<Number>::try_into(rterm) {
                match l * r {
                    Ok(Number::Integer(Int::Small(i))) => {
                        process.stack.store(self.dest, Term::Int(i).into());
                        Action::Continue
                    }
                    Ok(Number::Integer(Int::Big(i))) => {
                        let op = ops::LoadBig {
                            dest: self.dest,
                            value: i,
                        };
                        op.dispatch(emulator, process)
                    }
                    Ok(Number::Float(f)) => {
                        process.stack.store(self.dest, Term::Float(f).into());
                        Action::Continue
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = rhs;
                        emulator.handle_error(process)
                    }
                }
            } else {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                emulator.handle_error(process)
            }
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::Divide {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let lterm: Term = lhs.into();
        if let Ok(l) = TryInto::<Number>::try_into(lterm) {
            let rhs = process.stack.load(self.rhs);
            let rterm: Term = rhs.into();
            if let Ok(r) = TryInto::<Number>::try_into(rterm) {
                match l / r {
                    Ok(Number::Integer(Int::Small(i))) => {
                        process.stack.store(self.dest, Term::Int(i).into());
                        Action::Continue
                    }
                    Ok(Number::Integer(Int::Big(i))) => {
                        let op = ops::LoadBig {
                            dest: self.dest,
                            value: i,
                        };
                        op.dispatch(emulator, process)
                    }
                    Ok(Number::Float(f)) => {
                        process.stack.store(self.dest, Term::Float(f).into());
                        Action::Continue
                    }
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = rhs;
                        emulator.handle_error(process)
                    }
                }
            } else {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = rhs;
                emulator.handle_error(process)
            }
        } else {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = lhs;
            emulator.handle_error(process)
        }
    }
}
impl Inst for ops::ListAppend {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, _process: &mut ProcessLock) -> Action {
        todo!()
    }
}
impl Inst for ops::ListRemove {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, _process: &mut ProcessLock) -> Action {
        todo!()
    }
}
impl Inst for ops::IsEq {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if self.strict {
            let lhs = process.stack.load(self.lhs);
            let rhs = process.stack.load(self.rhs);
            process.stack.store(self.dest, lhs.exact_eq(&rhs).into());
        } else {
            let lhs: Term = process.stack.load(self.lhs).into();
            let rhs: Term = process.stack.load(self.rhs).into();
            process.stack.store(self.dest, (lhs == rhs).into());
        }
        Action::Continue
    }
}
impl Inst for ops::IsNeq {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        if self.strict {
            let lhs = process.stack.load(self.lhs);
            let rhs = process.stack.load(self.rhs);
            process.stack.store(self.dest, lhs.exact_ne(&rhs).into());
        } else {
            let lhs: Term = process.stack.load(self.lhs).into();
            let rhs: Term = process.stack.load(self.rhs).into();
            process.stack.store(self.dest, (lhs != rhs).into());
        }
        Action::Continue
    }
}
impl Inst for ops::IsGt {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        process.stack.store(self.dest, lhs.cmp(&rhs).is_gt().into());
        Action::Continue
    }
}
impl Inst for ops::IsGte {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        process.stack.store(self.dest, lhs.cmp(&rhs).is_ge().into());
        Action::Continue
    }
}
impl Inst for ops::IsLt {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        process.stack.store(self.dest, lhs.cmp(&rhs).is_lt().into());
        Action::Continue
    }
}
impl Inst for ops::IsLte {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let lhs = process.stack.load(self.lhs);
        let rhs = process.stack.load(self.rhs);
        process.stack.store(self.dest, lhs.cmp(&rhs).is_le().into());
        Action::Continue
    }
}
impl Inst for ops::Catch {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let cp = process.ip;
        process.stack.enter_catch(cp);
        // Skip over the LandingPad instruction, since that is only used as
        // the continuation for any exceptions which are raised.
        process.ip += 1;
        Action::Continue
    }
}
impl Inst for ops::EndCatch {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.stack.exit_catch();
        Action::Continue
    }
}
impl Inst for ops::LandingPad {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let class = process
            .exception_info
            .class()
            .map(|ec| ec.into())
            .unwrap_or(atoms::Error);
        let value = process.exception_info.value;
        // Store the raw trace pointer as an encoded term, the StackTrace instruction reifies it
        let trace = match process.exception_info.trace.as_ref() {
            None => OpaqueTerm::NONE,
            Some(arc) => OpaqueTerm::code(Arc::as_ptr(arc) as usize),
        };
        process.stack.store(self.kind, class.into());
        process.stack.store(self.reason, value.into());
        process.stack.store(self.trace, trace);
        // Jump to handler
        process.ip = process
            .ip
            .checked_add_signed(self.offset as isize - 1)
            .unwrap();
        Action::Continue
    }
}
impl Inst for ops::StackTrace {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // We've actually encoded the trace pointer on the stack, but it will still
        // be in the current exception state, so we just go straight to the source
        let term = match process.exception_info.trace.as_ref() {
            None => OpaqueTerm::NIL,
            Some(trace) => {
                let mut argv: Option<SmallVec<[OpaqueTerm; 8]>> = None;
                match process.exception_info.args {
                    Some(term) => match term.into() {
                        Term::Nil => {
                            argv = Some(Default::default());
                        }
                        Term::Cons(cons) => {
                            let mut args = SmallVec::<[OpaqueTerm; 8]>::default();
                            for maybe_improper in cons.iter_raw() {
                                match maybe_improper {
                                    Ok(t) => args.push(t),
                                    Err(t) => args.push(t),
                                }
                            }
                            argv = Some(args);
                        }
                        _ => {
                            argv = Some(smallvec![term]);
                        }
                    },
                    _ => (),
                }
                if let Ok(trace_term) = trace.as_term(argv.as_deref()) {
                    trace_term.into()
                } else {
                    return Action::Error(EmulatorError::SystemLimit);
                }
            }
        };
        process.stack.store(self.dest, term);
        Action::Continue
    }
}
impl Inst for ops::Error1 {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.value = process.stack.load(self.reason);
        process.exception_info.reason = match process.exception_info.value.into() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => atoms::Error.into(),
            },
            _ => atoms::Error.into(),
        };
        emulator.handle_error(process)
    }
}
impl Inst for ops::Throw1 {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.exception_info.flags = ExceptionFlags::THROW;
        let reason = process.stack.load(self.reason);
        process.exception_info.value = reason;
        process.exception_info.reason = match process.exception_info.value.into() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => atoms::Throw.into(),
            },
            _ => atoms::Throw.into(),
        };
        emulator.handle_error(process)
    }
}
impl Inst for ops::Raise {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let kind = process.stack.load(self.kind);
        process.exception_info.flags = match kind.into() {
            Term::Atom(a) if a == atoms::Throw => ExceptionFlags::THROW,
            Term::Atom(a) if a == atoms::Error => ExceptionFlags::ERROR,
            Term::Atom(a) if a == atoms::Exit => ExceptionFlags::EXIT,
            _ => {
                // Invalid kind, cancel exception and return badarg value
                process.stack.store(self.dest, atoms::Badarg.into());
                return Action::Continue;
            }
        };
        process.exception_info.value = process.stack.load(self.reason);
        let trace = self.trace.map(|t| process.stack.load(t).into());
        match trace {
            None | Some(Term::Nil) => {
                process.exception_info.trace = None;
            }
            Some(Term::Cons(cons)) => {
                if let Ok(trace) = emulator.parse_stacktrace(cons) {
                    process.exception_info.trace = Some(trace);
                } else {
                    // Invalid trace, cancel exception and return badarg value
                    process.stack.store(self.dest, atoms::Badarg.into());
                    return Action::Continue;
                }
            }
            Some(_) => {
                // Invalid trace, cancel exception and return badarg value
                process.stack.store(self.dest, atoms::Badarg.into());
                return Action::Continue;
            }
        }
        process.exception_info.reason = match process.exception_info.value.into() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => kind.as_atom().into(),
            },
            _ => kind.as_atom().into(),
        };
        emulator.handle_error(process)
    }
}
impl Inst for ops::SendOp {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let recipient_term = process.stack.load(self.recipient);
        match recipient_term.into() {
            Term::Pid(pid) => match registry::get_by_pid(pid.as_ref()) {
                None => Action::Continue,
                Some(recipient) => {
                    recipient
                        .send(
                            process.pid().into(),
                            process.stack.load(self.message).into(),
                        )
                        .ok();
                    Action::Continue
                }
            },
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = recipient_term;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::RecvPeek {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let available;
        let message;
        {
            let mut signals = process.signals().lock();
            available = signals.try_receive().is_ok();
            message = match signals.peek_message() {
                None => {
                    assert!(!available);
                    OpaqueTerm::NONE
                }
                Some(msg) => {
                    assert!(available);
                    // If this message is "accepted", then the underlying fragment
                    // will be added to the process off-heap fragment list, which
                    // will then be moved to the process heap during the next GC
                    msg.message.term
                }
            };
        }
        process.stack.store(self.available, available.into());
        process.stack.store(self.message, message);
        Action::Continue
    }
}
impl Inst for ops::RecvNext {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let mut signals = process.signals().lock();
        signals.next_message();
        Action::Continue
    }
}

impl Inst for ops::RecvWait {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // When control is transferred here, we have already checked the message queue
        // for messages, and none were found, so we're starting to wait for messages.
        //
        // If messages arrive, we must preserve the timer set up here (if one was set),
        // so that we timeout if received messages match none of the receive patterns.

        // Initialize the "timed out" result to false, it will be set true by the
        // RecvTimeout instruction if reached
        process.stack.store(self.dest, false.into());

        // Blocks execution until a message is received or `timeout` expires
        //
        // Control reaches this instruction when a previous recv_peek failed due to having
        // no messages in the queue, so we must suspend until woken up by receipt of a message
        // or until timeout.
        //
        // If it times out, control falls through to the next instruction which is the RecvTimeout
        // instruction. If a message is received it skips over it to the next instruction following
        // the timout.
        //
        // `dest` will be set to a boolean indicating whether or not the receive timed out
        // `timeout` is the timeout value to use, may either be the atom `infinity` or an integer
        // When this op is encountered, the ReceiveContext should have already been initialized by
        // RecvPeek
        let timeout_value = process.stack.load(self.timeout);
        let timeout = match timeout_value.into() {
            Term::Atom(a) if a == atoms::Infinity => Ok(Timeout::INFINITY),
            Term::Int(n) => n.try_into().map_err(|_| ()),
            _ => Err(()),
        };

        // Handle the unlikely case of an invalid timeout value
        if unlikely(timeout.is_err()) {
            // Invalid receive timeout, so cancel the receive attempt and raise a timeout_value
            // error
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::TimeoutValue.into();
            process.exception_info.value = timeout_value;
            return emulator.handle_error(process);
        }

        // If the timeout is infinite, no timer is used, the process is indefinitely suspended
        let timeout = timeout.unwrap();

        // On the initial recv_wait_timeout, no timer is set, but on subsequent visits,
        // it may be, so we need to first determine if we have an active timer, if not
        // we can set one up, otherwise we need to determine if the timer is still active
        // and handle it appropriately
        match process.timer() {
            ProcessTimer::Active(_) => {
                // We have an active timer already, so we jump straight to suspending
                process.set_status_flags(StatusFlags::SUSPENDED, Ordering::Release);
                Action::Suspend
            }
            ProcessTimer::TimedOut => {
                // Let the RecvTimeout instruction handle this
                Action::Continue
            }
            ProcessTimer::None if timeout == Timeout::INFINITY => {
                // Update the process flags to indicate that this process is no longer active and is
                // suspended
                process.set_status_flags(StatusFlags::SUSPENDED, Ordering::Release);
                // The process will go back in the scheduler queue, but won't do anything but handle
                // signals until a message is received.
                Action::Suspend
            }
            ProcessTimer::None => {
                // Otherwise, we register a new timeout timer for this process, and suspend until
                // either the timeout wakes the process, or we are rescheduled due
                // to receipt of a message
                match emulator.timeout_after(process, timeout) {
                    Ok(_) => {
                        process.set_status_flags(StatusFlags::SUSPENDED, Ordering::Release);
                        process.set_flags(ProcessFlags::IN_TIMER_QUEUE);
                        Action::Suspend
                    }
                    // The timeout would expire immediately, so don't bother suspending
                    Err(TimerError::Expired(_)) => {
                        process.stack.store(self.dest, true.into());
                        // Skip over the RecvTimeout instruction
                        process.ip += 1;
                        Action::Continue
                    }
                    Err(TimerError::InvalidTimeout(_)) | Err(TimerError::Alloc) => unreachable!(),
                }
            }
        }
    }
}
impl Inst for ops::RecvTimeout {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // We reach here when rescheduled after a RecvWait instruction, and we must update the
        // process based on how we were rescheduled
        let timed_out = process.flags.contains(ProcessFlags::TIMEOUT);
        process.stack.store(self.dest, timed_out.into());
        Action::Continue
    }
}
impl Inst for ops::RecvPop {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        // If we have a timeout associated with the current receive, cancel it
        if let (ProcessTimer::Active(_), Some(timer_ref)) = process.cancel_timer() {
            emulator.cancel_timer(timer_ref).ok();
        }

        // Pop the message off the queue and drop it
        let mut signals = process.signals().lock();
        let mut message = signals.remove_message();
        drop(signals);
        if let Some(fragment_ptr) = message.message.fragment.take() {
            unsafe {
                process
                    .heap_fragments
                    .push_back(UnsafeRef::from_raw(fragment_ptr.as_ptr().cast_const()));
            }
        }
        Action::Continue
    }
}

impl Inst for ops::Await {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_rt::process::GeneratorState;

        // We should have a generator ready to poll
        let mut generator = process.awaiting.take().unwrap();
        match generator.resume(process) {
            GeneratorState::Yielded(_) => {
                // Yield back to the scheduler for now
                //
                // NOTE: We do not mark the process as suspended currently because
                // we are still ironing out how to best implement a Waker for arbitrary
                // NIFs/BIFs. For now, we consider the overhead of scheduling a process
                // to poll a generator that is just going to yield cheap enough that it
                // isn't important.
                process.awaiting = Some(generator);
                Action::Yield
            }
            GeneratorState::Completed(Ok(result)) => {
                process.stack.store(RETURN_REG, result.into());
                let op = ops::Ret { reg: RETURN_REG };
                op.dispatch(emulator, process)
            }
            GeneratorState::Completed(Err(_)) => emulator.handle_error(process),
        }
    }
}
impl Inst for ops::Trap {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let op = ops::EnterStatic {
            callee: process.trap.take().unwrap(),
        };
        op.dispatch(emulator, process)
    }
}
impl Inst for ops::Yield {
    #[inline(always)]
    fn dispatch(&self, _emulator: &Emulator, _process: &mut ProcessLock) -> Action {
        Action::Yield
    }
}
impl Inst for ops::GarbageCollect {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        match gc::garbage_collect(process, Default::default()) {
            Ok(_) => {
                if process.reductions >= MAX_REDUCTIONS {
                    // Make sure the process is marked as ACTIVE before yielding
                    process.set_status_flags(StatusFlags::ACTIVE, Ordering::Release);
                    Action::Yield
                } else {
                    Action::Continue
                }
            }
            Err(_) => emulator.handle_error(process),
        }
    }
}
impl Inst for ops::NormalExit {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        process.ip = CONTINUE_EXIT_IP;
        process.stack.nocatch();
        process.exception_info.flags = ExceptionFlags::EXIT & !(ExceptionFlags::SAVETRACE);
        process.exception_info.reason = atoms::Normal.into();
        process.exception_info.value = atoms::Normal.into();
        process.exception_info.trace = None;
        emulator.terminate_process(process, atoms::Normal.into())
    }
}
impl Inst for ops::ContinueExit {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let mut status = process.status(Ordering::Relaxed);
        assert!(status.contains(StatusFlags::EXITING));
        process.ip = CONTINUE_EXIT_IP;

        'outer: loop {
            match process.continue_exit {
                ContinueExitPhase::Timers => {
                    // if process.bif_timers {
                    //     cost = erts_cancel_bif_timers(process, process.bif_timers,
                    // process.reductions);     process.reductions += cost;
                    //     if process.reductions >= MAX_REDUCTIONS { Action::Yield }
                    //     process.bif_timers = null;
                    // }
                    process.continue_exit = ContinueExitPhase::UsingDb;
                }
                ContinueExitPhase::UsingDb => {
                    if process.flags.contains(ProcessFlags::USING_DB) {
                        //if erts_db_process_exiting(process) {
                        //    // yield
                        //    break;
                        //}
                        //process.flags.remove(ProcessFlags::USING_DB);
                        todo!("erts_db_process_exiting");
                    }
                    process.continue_exit = ContinueExitPhase::CleanSysTasks;
                }
                ContinueExitPhase::CleanSysTasks => {
                    process.flags.remove(ProcessFlags::DISABLE_GC);
                    // check if there are delayed gc tasks and move them to front of sys task queue
                    // if there are, unset StatusFlags::DELAYED_SYS and set StatusFlags::ACTIVE_SYS
                    // | StatusFlags::SYS_TASKS
                    process.continue_exit = ContinueExitPhase::Free;
                }
                ContinueExitPhase::Free => {
                    if process.flags.contains(ProcessFlags::USING_DDLL) {
                        // process.flags.remove(ProcessFlags::USING_DDLL);
                        todo!("erts_ddll_proc_dead");
                    }

                    // This is the point at which the process is actually dead
                    registry::unregister_process(process.id()).unwrap();

                    // All erlang resources have too be deallocated before this point,
                    // e.g. registered name, so monitoring and linked processes can be
                    // sure that all interesting resources have been deallocated when the
                    // monitors and/or links hit.
                    process.set_status_flags(StatusFlags::FREE, Ordering::Release);

                    // if process.flags.contains(ProcessFlags::DISTRIBUTION) {
                    //   state.dep = process.distribution.take();
                    // }
                    process.reductions += 50;
                    process.continue_exit = ContinueExitPhase::CleanSysTasksAfter;
                }
                ContinueExitPhase::CleanSysTasksAfter => {
                    status = process.status(Ordering::Acquire);
                    if status.contains(StatusFlags::SYS_TASKS) {
                        emulator.cleanup_sys_tasks(process);
                        if process.reductions >= MAX_REDUCTIONS {
                            // yield
                            break;
                        }
                    }

                    // process.sys_tasks.clear();

                    // if let Some(ref mut dist) = state.dep {
                    //     let reason = if state.reason == atoms::Kill.into() {
                    //         atoms::Killed.into()
                    //     } else {
                    //         state.reason
                    //     };
                    //     erts_do_net_exits(dist, reason);
                    //     erts_deref_dist_entry(dist);
                    // }

                    // Disable GC so that reason does not get moved
                    process.flags |= ProcessFlags::DISABLE_GC;

                    {
                        let mut sigq = process.signals().lock();
                        loop {
                            let mut new_status = status;
                            sigq.flush_buffers();
                            if sigq.is_empty() {
                                new_status.remove(
                                    StatusFlags::HAS_IN_TRANSIT_SIGNALS
                                        | StatusFlags::HAS_PENDING_SIGNALS
                                        | StatusFlags::MAYBE_SELF_SIGNALS,
                                );
                            } else {
                                new_status.remove(StatusFlags::HAS_IN_TRANSIT_SIGNALS);
                                new_status |= StatusFlags::HAS_PENDING_SIGNALS;
                            }
                            match process.cmpxchg_status_flags(status, new_status) {
                                Ok(_) => {
                                    break;
                                }
                                Err(current) => {
                                    status = current;
                                }
                            }
                        }
                    }
                    process.continue_exit = ContinueExitPhase::Links;
                }
                ContinueExitPhase::Links => {
                    let mut links = process.links.take();
                    for (_, linked) in links.drain() {
                        emulator.handle_exit_link(process, process.exception_info.value, linked);
                    }
                    process.continue_exit = ContinueExitPhase::Monitors;
                }
                ContinueExitPhase::Monitors => {
                    let mut monitors = process.monitored_by.take();
                    while let Some(monitored_by) = monitors.pop_front() {
                        emulator.handle_exit_monitor(
                            process,
                            process.exception_info.value,
                            monitored_by,
                        );
                    }
                    let mut monitors = process.monitored.take();
                    let mut cursor = monitors.front_mut();
                    while let Some(monitored) = cursor.remove() {
                        emulator.handle_exit_demonitor(
                            process,
                            process.exception_info.value,
                            monitored,
                        );
                    }
                    process.continue_exit = ContinueExitPhase::HandleProcessSignals;
                }
                ContinueExitPhase::HandleProcessSignals => {
                    let limit = MAX_REDUCTIONS.saturating_sub(process.reductions)
                        * ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR;
                    let mut count = 0;
                    let proc = process.strong();
                    let mut sigq = proc.signals().lock();
                    while let Some(entry) = sigq.pop() {
                        count += 1;
                        if count >= limit {
                            break 'outer;
                        }
                        match entry.signal {
                            Signal::Message(_)
                            | Signal::Exit(_)
                            | Signal::ExitLink(_)
                            | Signal::MonitorDown(_)
                            | Signal::Demonitor(_)
                            | Signal::UnlinkAck(_) => continue,
                            Signal::Monitor(sig) => {
                                emulator.handle_exit_monitor(
                                    process,
                                    atoms::Noproc.into(),
                                    sig.monitor,
                                );
                                count += 4;
                            }
                            Signal::Link(sig) => {
                                emulator.handle_exit_link(process, atoms::Noproc.into(), sig.link);
                                count += 1;
                            }
                            Signal::Unlink(sig) => {
                                count += emulator.handle_unlink(process, sig.sender, sig.id);
                            }
                            Signal::GroupLeader(sig) => {
                                let sender = WeakAddress::Process(sig.sender);
                                if let Some(Registrant::Process(p)) = sender.try_resolve() {
                                    emulator.send_group_leader_reply(p, sig.reference, false);
                                }
                            }
                            Signal::IsAlive(sig) => {
                                let sender = WeakAddress::Process(sig.sender);
                                if let Some(Registrant::Process(p)) = sender.try_resolve() {
                                    emulator.send_is_alive_reply(p, sig.reference, false);
                                }
                            }
                            Signal::ProcessInfo(sig) => {
                                emulator.handle_process_info_signal(process, sig, false);
                            }
                            Signal::Flush(_sig) => {
                                assert!(sigq.flags().contains(SignalQueueFlags::FLUSHING));
                                sigq.set_flags(SignalQueueFlags::FLUSHED);
                            }
                            Signal::Rpc(sig) => {
                                count += emulator.handle_rpc(process, sig);
                            }
                        }
                    }
                    process.reductions += 1 + (count / ERTS_SIGNAL_REDUCTIONS_COUNT_FACTOR);
                    process
                        .remove_status_flags(StatusFlags::HAS_PENDING_SIGNALS, Ordering::Relaxed);
                    process.continue_exit = ContinueExitPhase::DistSend;
                }
                ContinueExitPhase::DistSend => {
                    process.continue_exit = ContinueExitPhase::DistLinks;
                }
                ContinueExitPhase::DistLinks => {
                    process.continue_exit = ContinueExitPhase::DistMonitors;
                }
                ContinueExitPhase::DistMonitors => {
                    process.continue_exit = ContinueExitPhase::PendingSpawnMonitors;
                }
                ContinueExitPhase::PendingSpawnMonitors => {
                    process.continue_exit = ContinueExitPhase::Done;
                }
                ContinueExitPhase::Done => {
                    // From this point onwards we are no longer allowed to yield and this
                    // process is being deallocated (once all the arcs get dropped)
                    process.flags.remove(ProcessFlags::DISABLE_GC);
                    status = process.status(Ordering::Relaxed);
                    loop {
                        let mut new_status = status;
                        new_status.remove(StatusFlags::ACTIVE | StatusFlags::ACTIVE_SYS);
                        match process.cmpxchg_status_flags(status, new_status) {
                            Ok(_current) => {
                                break;
                            }
                            Err(current) => {
                                status = current;
                            }
                        }
                    }
                    return Action::Killed;
                }
            }
        }

        Action::Yield
    }
}
impl Inst for ops::Exit1 {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let reason = process.stack.load(self.reason);
        process.exception_info.flags = ExceptionFlags::EXIT;
        process.exception_info.reason = match reason.into() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => atoms::Exit.into(),
            },
            _ => atoms::Exit.into(),
        };
        process.exception_info.value = reason;
        emulator.handle_error(process)
    }
}
impl Inst for ops::Exit2 {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let pid = process.stack.load(self.pid);
        let reason = process.stack.load(self.reason);
        match pid.into() {
            Term::Pid(boxed) => {
                let receiver_id = boxed.id();
                let is_exiting_self = process.id() == receiver_id;
                if is_exiting_self {
                    let is_self_trapping_exits = process.flags.contains(ProcessFlags::TRAP_EXIT);
                    let is_suicide = reason == atoms::Kill || !is_self_trapping_exits;
                    process
                        .send_signal(SignalEntry::new(Signal::Exit(signals::Exit {
                            sender: Some(process.addr()),
                            reason: TermFragment::new(reason.into()).unwrap(),
                            normal_kills: is_suicide,
                        })))
                        .ok();
                    // Force a yield to handle pending signals immediately
                    Action::Yield
                } else {
                    if let Some(receiver) = registry::get_by_pid(boxed.as_ref()) {
                        receiver
                            .send_signal(SignalEntry::new(Signal::Exit(signals::Exit {
                                sender: Some(process.addr()),
                                reason: TermFragment::new(reason.into()).unwrap(),
                                normal_kills: false,
                            })))
                            .ok();
                    }
                    Action::Continue
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = pid;
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::Halt {
    #[inline(never)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let status = process.stack.load(self.status);
        match status.into() {
            Term::Atom(status) if status == atoms::Abort => {
                std::process::abort();
            }
            Term::Int(status) if status >= 0 => {
                Action::Error(EmulatorError::Halt(status.try_into().unwrap_or(1)))
            }
            other => {
                if let Some(string) = other.as_bitstring() {
                    if let Some(slogan) = string.as_str() {
                        // TODO: crash dump
                        eprintln!("{}", slogan);
                    }
                    Action::Error(EmulatorError::Halt(1))
                } else {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Badarg.into();
                    process.exception_info.value = status;
                    emulator.handle_error(process)
                }
            }
        }
    }
}
impl Inst for ops::BsInit {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_binary::BitVec;
        let buffer = Box::new(BitVec::new());
        let term = OpaqueTerm::code(Box::into_raw(buffer) as usize);
        process.stack.store(self.dest, term);
        Action::Continue
    }
}
impl Inst for ops::BsPush {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_binary::{BinaryEntrySpecifier, BitVec, Bitstring};
        use firefly_number::{f16, ToPrimitive};

        let builder = process.stack.load(self.builder);
        assert!(builder.is_code());
        let mut ptr = unsafe { NonNull::new_unchecked(builder.as_code() as *mut BitVec) };
        let buffer = unsafe { ptr.as_mut() };
        let value = process.stack.load(self.value);

        // Process/validate the optional size argument
        let size_term = self.size.map(|sz| process.stack.load(sz));
        let size: Option<usize> = match size_term {
            None => None,
            Some(sz) => match sz.into() {
                Term::Int(i) => match i.try_into() {
                    Ok(n) => Some(n),
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = sz;
                        return emulator.handle_error(process);
                    }
                },
                _ => {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Badarg.into();
                    process.exception_info.value = sz;
                    return emulator.handle_error(process);
                }
            },
        };

        match self.spec {
            BinaryEntrySpecifier::Integer {
                signed,
                unit,
                endianness,
            } => {
                // Size MUST be a non-negative integer
                let size = size.unwrap();
                let num_bits = size * (unit as usize);
                match value.into() {
                    // Pushing with a size of zero has no effect
                    Term::Int(_) | Term::BigInt(_) if num_bits == 0 => (),
                    Term::Int(i) if signed => buffer.push_ap_number(i, num_bits, endianness),
                    Term::Int(i) => buffer.push_ap_number(i as u64, num_bits, endianness),
                    Term::BigInt(i) => {
                        buffer.push_ap_bigint(i.deref(), num_bits, signed, endianness)
                    }
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                }
            }
            BinaryEntrySpecifier::Float { unit, endianness } => {
                // Size MUST be one of 16, 32, 64
                let size = size.unwrap();
                match size * unit as usize {
                    16 | 32 | 64 => (),
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = size_term.unwrap();
                        return emulator.handle_error(process);
                    }
                }
                // Value can be any integer or float
                match value.into() {
                    Term::Float(f) if size == 16 => {
                        buffer.push_number(f16::from_f64(f.inner()), endianness)
                    }
                    Term::Float(f) if size == 32 => {
                        buffer.push_number(f.inner() as f32, endianness)
                    }
                    Term::Float(f) if size == 64 => buffer.push_number(f.inner(), endianness),
                    Term::Int(i) if size == 16 => {
                        buffer.push_number(f16::from_f64(i as f64), endianness)
                    }
                    Term::Int(i) if size == 32 => buffer.push_number(i as f32, endianness),
                    Term::Int(i) if size == 64 => buffer.push_number(i as f64, endianness),
                    Term::BigInt(i) => match i.to_f64() {
                        Some(f) if size == 16 => buffer.push_number(f16::from_f64(f), endianness),
                        Some(f) if size == 32 => buffer.push_number(f as f32, endianness),
                        Some(f) if size == 64 => buffer.push_number(f, endianness),
                        _ => {
                            process.exception_info.flags = ExceptionFlags::ERROR;
                            process.exception_info.reason = atoms::Badarg.into();
                            process.exception_info.value = value;
                            return emulator.handle_error(process);
                        }
                    },
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                }
            }
            BinaryEntrySpecifier::Binary { unit } => {
                // Size must be a non-negative integer, or None to represent pushing all of the
                // source value into the destination buffer Term must be a
                // bitstring/binary
                let term: Term = value.into();
                match term.as_bitstring() {
                    // Push all of a binary
                    Some(bs) if unit == 8 && size.is_none() => {
                        // The source value must be a binary
                        if !bs.is_binary() {
                            process.exception_info.flags = ExceptionFlags::ERROR;
                            process.exception_info.reason = atoms::Badarg.into();
                            process.exception_info.value = value;
                            return emulator.handle_error(process);
                        }
                        buffer.extend(bs.bytes());
                    }
                    // Push `size` bytes of the given bitstring
                    Some(bs) if unit == 8 => {
                        // Selects precisely `size` bytes from the underlying data
                        // The value must be at least as large as the requested size
                        match bs.select_bytes(size.unwrap()) {
                            Ok(selection) => {
                                buffer.extend(selection.bytes());
                            }
                            _ => {
                                process.exception_info.flags = ExceptionFlags::ERROR;
                                process.exception_info.reason = atoms::Badarg.into();
                                process.exception_info.value = value;
                                return emulator.handle_error(process);
                            }
                        }
                    }
                    // Push all bits of a bitstring
                    Some(bs) if size.is_none() => {
                        // The source value may be either a binary or bitstring
                        if bs.is_binary() {
                            buffer.extend(bs.bytes());
                        } else {
                            buffer.extend(bs.bits());
                        }
                    }
                    // Push `size * unit` bits of the given bitstring
                    Some(bs) => {
                        let size = size.unwrap();
                        let bitsize = size * (unit as usize);
                        // Selects precisely `bitsize` bits from the underlying data
                        // The value must be at least as large as the requested size
                        match bs.select_bits(bitsize) {
                            Ok(selection) => {
                                buffer.extend(selection.bits());
                            }
                            _ => {
                                process.exception_info.flags = ExceptionFlags::ERROR;
                                process.exception_info.reason = atoms::Badarg.into();
                                process.exception_info.value = value;
                                return emulator.handle_error(process);
                            }
                        }
                    }
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                }
            }
            BinaryEntrySpecifier::Utf8 => {
                let c: char = match value.try_into() {
                    Ok(c) => c,
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                };
                buffer.push_utf8(c);
            }
            BinaryEntrySpecifier::Utf16 { endianness } => {
                let c: char = match value.try_into() {
                    Ok(c) => c,
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                };
                buffer.push_utf16(c, endianness);
            }
            BinaryEntrySpecifier::Utf32 { endianness } => {
                let c: char = match value.try_into() {
                    Ok(c) => c,
                    _ => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = value;
                        return emulator.handle_error(process);
                    }
                };
                buffer.push_utf32(c, endianness);
            }
        }

        process.stack.store(self.dest, builder);
        Action::Continue
    }
}
impl Inst for ops::BsFinish {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_binary::{BitVec, Bitstring};

        let builder = process.stack.load(self.builder);
        assert!(builder.is_code());
        let ptr = unsafe { NonNull::new_unchecked(builder.as_code() as *mut BitVec) };
        let buffer = unsafe { ptr.as_ref() };
        match buffer.byte_size() {
            n if n <= 64 => {
                let bytes = unsafe { buffer.as_bytes_unchecked() };
                if let Ok(bin) = BinaryData::from_small_bytes(bytes, process) {
                    // Release the buffer
                    drop(unsafe { Box::from_raw(ptr.as_ptr()) });
                    process.stack.store(self.dest, bin.into());
                    Action::Continue
                } else {
                    let mut builder = LayoutBuilder::new();
                    builder.build_heap_binary(n);
                    process.gc_needed = builder.finish().size();
                    process.ip -= 1;
                    GC.dispatch(emulator, process)
                }
            }
            _ => {
                let bytes = unsafe { buffer.as_bytes_unchecked() };
                let bin = BinaryData::from_bytes(bytes);
                // Release the buffer
                drop(unsafe { Box::from_raw(ptr.as_ptr()) });
                process.stack.store(self.dest, bin.into());
                Action::Continue
            }
        }
    }
}
impl Inst for ops::BsMatchStart {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let bin = process.stack.load(self.bin);
        if let TermType::Binary = bin.r#typeof() {
            match MatchContext::new(bin, process) {
                Ok(context) => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    process.stack.store(self.context, context.into());
                    Action::Continue
                }
                Err(_) => {
                    process.gc_needed = mem::size_of::<MatchContext>();
                    process.ip -= 1;
                    GC.dispatch(emulator, process)
                }
            }
        } else {
            process.stack.store(self.is_err, OpaqueTerm::TRUE);
            process.stack.store(self.context, OpaqueTerm::NONE);
            Action::Continue
        }
    }
}
impl Inst for ops::BsMatch {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_binary::BinaryEntrySpecifier;
        use firefly_number::f16;

        let heap_top = process.heap.heap_top() as usize;
        // We perform matching with a stack copy of the context since we don't want to mutate the
        // original If the match is unsuccessful, we simply discard it; but if it is
        // successful, we store it on the process heap and use it for subsequent dependent
        // matching operations
        let context_term = process.stack.load(self.context);
        assert!(context_term.is_match_context());
        let original_context = unsafe { Gc::from_raw(context_term.as_ptr() as *mut MatchContext) };
        let mut context = original_context.as_ref().clone();
        let matcher = context.matcher_mut();
        let size_term = self.size.map(|sz| process.stack.load(sz));
        let size: Option<usize> = match size_term {
            None => None,
            Some(sz) => match sz.into() {
                Term::Int(i) => match i.try_into() {
                    Ok(i) => Some(i),
                    Err(_) => {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = sz;
                        return emulator.handle_error(process);
                    }
                },
                _ => {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Badarg.into();
                    process.exception_info.value = sz;
                    return emulator.handle_error(process);
                }
            },
        };

        match self.spec {
            BinaryEntrySpecifier::Integer {
                signed,
                unit,
                endianness,
            } => {
                let size = size.unwrap();
                let bitsize = unit as usize * size;
                if bitsize == 0 {
                    // Since we we haven't modified the context, we can simply return the original
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    process.stack.store(self.value, OpaqueTerm::ZERO);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                } else if (signed && bitsize > OpaqueTerm::INT_BITSIZE)
                    || (!signed && bitsize >= OpaqueTerm::INT_BITSIZE)
                {
                    match matcher.match_bigint(bitsize, signed, endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(i) => match Gc::new_in(i, process) {
                            Ok(big) => {
                                process.stack.store(self.is_err, OpaqueTerm::FALSE);
                                process.stack.store(self.value, big.into());
                                match clone_context_to_heap(context, process) {
                                    Ok(next) => {
                                        process.stack.store(self.next, next);
                                        Action::Continue
                                    }
                                    Err(_) => {
                                        let extra = process.heap.heap_top() as usize - heap_top;
                                        process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                        process.ip -= 1;
                                        GC.dispatch(emulator, process)
                                    }
                                }
                            }
                            Err(_) => {
                                let extra = process.heap.heap_top() as usize - heap_top;
                                process.gc_needed = mem::size_of::<MatchContext>()
                                    + extra
                                    + mem::size_of::<BigInt>();
                                process.ip -= 1;
                                GC.dispatch(emulator, process)
                            }
                        },
                    }
                } else if signed {
                    match matcher.match_ap_number::<i64, 8>(bitsize, endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(i) => {
                            process.stack.store(self.is_err, OpaqueTerm::FALSE);
                            process.stack.store(self.value, Term::Int(i).into());
                            match clone_context_to_heap(context, process) {
                                Ok(next) => {
                                    process.stack.store(self.next, next);
                                    Action::Continue
                                }
                                Err(_) => {
                                    let extra = process.heap.heap_top() as usize - heap_top;
                                    process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                    process.ip -= 1;
                                    GC.dispatch(emulator, process)
                                }
                            }
                        }
                    }
                } else {
                    match matcher.match_ap_number::<u64, 8>(bitsize, endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(i) => {
                            process.stack.store(self.is_err, OpaqueTerm::FALSE);
                            process.stack.store(self.value, Term::Int(i as i64).into());
                            match clone_context_to_heap(context, process) {
                                Ok(next) => {
                                    process.stack.store(self.next, next);
                                    Action::Continue
                                }
                                Err(_) => {
                                    let extra = process.heap.heap_top() as usize - heap_top;
                                    process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                    process.ip -= 1;
                                    GC.dispatch(emulator, process)
                                }
                            }
                        }
                    }
                }
            }
            BinaryEntrySpecifier::Float { unit, endianness } => {
                let size = size.unwrap();
                let bitsize = unit as usize * size;
                match bitsize {
                    // No clone needed as we haven't modified the original
                    0 => {
                        process.stack.store(self.is_err, OpaqueTerm::FALSE);
                        process.stack.store(self.value, 0.0f64.into());
                        process.stack.store(self.next, context_term);
                        Action::Continue
                    }
                    16 => match matcher.match_number::<f16, 2>(endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(n) => {
                            let f: f64 = n.into();
                            process.stack.store(self.is_err, OpaqueTerm::FALSE);
                            process.stack.store(self.value, f.into());
                            match clone_context_to_heap(context, process) {
                                Ok(next) => {
                                    process.stack.store(self.next, next);
                                    Action::Continue
                                }
                                Err(_) => {
                                    let extra = process.heap.heap_top() as usize - heap_top;
                                    process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                    process.ip -= 1;
                                    GC.dispatch(emulator, process)
                                }
                            }
                        }
                    },
                    32 => match matcher.match_number::<f32, 4>(endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(n) => {
                            let f: f64 = n.into();
                            process.stack.store(self.is_err, OpaqueTerm::FALSE);
                            process.stack.store(self.value, f.into());
                            match clone_context_to_heap(context, process) {
                                Ok(next) => {
                                    process.stack.store(self.next, next);
                                    Action::Continue
                                }
                                Err(_) => {
                                    let extra = process.heap.heap_top() as usize - heap_top;
                                    process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                    process.ip -= 1;
                                    GC.dispatch(emulator, process)
                                }
                            }
                        }
                    },
                    64 => match matcher.match_number::<f64, 8>(endianness) {
                        None => {
                            process.stack.store(self.is_err, OpaqueTerm::TRUE);
                            process.stack.store(self.value, OpaqueTerm::NONE);
                            process.stack.store(self.next, context_term);
                            Action::Continue
                        }
                        Some(f) => {
                            process.stack.store(self.is_err, OpaqueTerm::FALSE);
                            process.stack.store(self.value, f.into());
                            match clone_context_to_heap(context, process) {
                                Ok(next) => {
                                    process.stack.store(self.next, next);
                                    Action::Continue
                                }
                                Err(_) => {
                                    let extra = process.heap.heap_top() as usize - heap_top;
                                    process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                    process.ip -= 1;
                                    GC.dispatch(emulator, process)
                                }
                            }
                        }
                    },
                    bitsize => panic!(
                        "invalid bitsize for floats, must be 16, 32 or 64, got {}",
                        bitsize
                    ),
                }
            }
            BinaryEntrySpecifier::Binary { unit } => {
                match size {
                    Some(size) => {
                        // Match `size * unit` bits of the binary/bitstring
                        let bitsize = unit as usize * size;
                        match matcher.match_bits(bitsize) {
                            None => {
                                process.stack.store(self.is_err, OpaqueTerm::TRUE);
                                process.stack.store(self.value, OpaqueTerm::NONE);
                                process.stack.store(self.next, context_term);
                                Action::Continue
                            }
                            Some(selection) => {
                                match Gc::new_in(
                                    BitSlice::from_selection(context.owner(), selection),
                                    process,
                                ) {
                                    Ok(bin) => {
                                        process.stack.store(self.is_err, OpaqueTerm::FALSE);
                                        process.stack.store(self.value, bin.into());
                                        match clone_context_to_heap(context, process) {
                                            Ok(next) => {
                                                process.stack.store(self.next, next);
                                                Action::Continue
                                            }
                                            Err(_) => {
                                                let extra =
                                                    process.heap.heap_top() as usize - heap_top;
                                                process.gc_needed =
                                                    mem::size_of::<MatchContext>() + extra;
                                                process.ip -= 1;
                                                GC.dispatch(emulator, process)
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        let extra = process.heap.heap_top() as usize - heap_top;
                                        process.gc_needed = mem::size_of::<MatchContext>()
                                            + extra
                                            + mem::size_of::<BitSlice>();
                                        process.ip -= 1;
                                        GC.dispatch(emulator, process)
                                    }
                                }
                            }
                        }
                    }
                    None if unit == 8 => {
                        // Match the remaining bits, as long as those bits form a binary
                        match matcher.match_binary() {
                            None => {
                                process.stack.store(self.is_err, OpaqueTerm::TRUE);
                                process.stack.store(self.value, OpaqueTerm::NONE);
                                process.stack.store(self.next, context_term);
                                Action::Continue
                            }
                            Some(selection) => {
                                match Gc::new_in(
                                    BitSlice::from_selection(context.owner(), selection),
                                    process,
                                ) {
                                    Ok(bin) => {
                                        process.stack.store(self.is_err, OpaqueTerm::FALSE);
                                        process.stack.store(self.value, bin.into());
                                        match clone_context_to_heap(context, process) {
                                            Ok(next) => {
                                                process.stack.store(self.next, next);
                                                Action::Continue
                                            }
                                            Err(_) => {
                                                let extra =
                                                    process.heap.heap_top() as usize - heap_top;
                                                process.gc_needed =
                                                    mem::size_of::<MatchContext>() + extra;
                                                process.ip -= 1;
                                                GC.dispatch(emulator, process)
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        let extra = process.heap.heap_top() as usize - heap_top;
                                        process.gc_needed = mem::size_of::<MatchContext>()
                                            + extra
                                            + mem::size_of::<BitSlice>();
                                        process.ip -= 1;
                                        GC.dispatch(emulator, process)
                                    }
                                }
                            }
                        }
                    }
                    None => {
                        // Match the remaining bits
                        let selection = matcher.match_any();
                        match Gc::new_in(
                            BitSlice::from_selection(context.owner(), selection),
                            process,
                        ) {
                            Ok(bin) => {
                                process.stack.store(self.is_err, OpaqueTerm::FALSE);
                                process.stack.store(self.value, bin.into());
                                match clone_context_to_heap(context, process) {
                                    Ok(next) => {
                                        process.stack.store(self.next, next);
                                        Action::Continue
                                    }
                                    Err(_) => {
                                        let extra = process.heap.heap_top() as usize - heap_top;
                                        process.gc_needed = mem::size_of::<MatchContext>() + extra;
                                        process.ip -= 1;
                                        GC.dispatch(emulator, process)
                                    }
                                }
                            }
                            Err(_) => {
                                let extra = process.heap.heap_top() as usize - heap_top;
                                process.gc_needed = mem::size_of::<MatchContext>()
                                    + extra
                                    + mem::size_of::<BitSlice>();
                                process.ip -= 1;
                                GC.dispatch(emulator, process)
                            }
                        }
                    }
                }
            }
            BinaryEntrySpecifier::Utf8 => match matcher.match_utf8() {
                None => {
                    process.stack.store(self.is_err, OpaqueTerm::TRUE);
                    process.stack.store(self.value, OpaqueTerm::NONE);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                }
                Some(c) => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    process.stack.store(self.value, Term::Int(c as i64).into());
                    match clone_context_to_heap(context, process) {
                        Ok(next) => {
                            process.stack.store(self.next, next);
                            Action::Continue
                        }
                        Err(_) => {
                            let extra = process.heap.heap_top() as usize - heap_top;
                            process.gc_needed = mem::size_of::<MatchContext>() + extra;
                            process.ip -= 1;
                            GC.dispatch(emulator, process)
                        }
                    }
                }
            },
            BinaryEntrySpecifier::Utf16 { endianness } => match matcher.match_utf16(endianness) {
                None => {
                    process.stack.store(self.is_err, OpaqueTerm::TRUE);
                    process.stack.store(self.value, OpaqueTerm::NONE);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                }
                Some(c) => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    process.stack.store(self.value, Term::Int(c as i64).into());
                    match clone_context_to_heap(context, process) {
                        Ok(next) => {
                            process.stack.store(self.next, next);
                            Action::Continue
                        }
                        Err(_) => {
                            let extra = process.heap.heap_top() as usize - heap_top;
                            process.gc_needed = mem::size_of::<MatchContext>() + extra;
                            process.ip -= 1;
                            GC.dispatch(emulator, process)
                        }
                    }
                }
            },
            BinaryEntrySpecifier::Utf32 { endianness } => match matcher.match_utf32(endianness) {
                None => {
                    process.stack.store(self.is_err, OpaqueTerm::TRUE);
                    process.stack.store(self.value, OpaqueTerm::NONE);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                }
                Some(c) => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    process.stack.store(self.value, Term::Int(c as i64).into());
                    match clone_context_to_heap(context, process) {
                        Ok(next) => {
                            process.stack.store(self.next, next);
                            Action::Continue
                        }
                        Err(_) => {
                            let extra = process.heap.heap_top() as usize - heap_top;
                            process.gc_needed = mem::size_of::<MatchContext>() + extra;
                            process.ip -= 1;
                            GC.dispatch(emulator, process)
                        }
                    }
                }
            },
        }
    }
}
impl Inst for ops::BsMatchSkip {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        use firefly_binary::Endianness;
        use firefly_bytecode::ops::BsMatchSkipType;

        let heap_top = process.heap.heap_top() as usize;
        let context_term = process.stack.load(self.context);
        assert!(context_term.is_match_context());
        let original_context = unsafe { Gc::from_raw(context_term.as_ptr() as *mut MatchContext) };
        let mut context = original_context.as_ref().clone();
        let matcher = context.matcher_mut();
        let size_term = process.stack.load(self.size);
        let size: usize = match size_term.into() {
            Term::Int(i) => match i.try_into() {
                Ok(i) => i,
                Err(_) => {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Badarg.into();
                    process.exception_info.value = size_term;
                    return emulator.handle_error(process);
                }
            },
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = size_term;
                return emulator.handle_error(process);
            }
        };
        let Term::Int(value) = process.stack.load(self.value).into() else { panic!("invalid argument to bs_match intrinsic") };

        let bitsize = self.unit as usize * size;
        if bitsize == 0 {
            process.stack.store(self.is_err, OpaqueTerm::FALSE);
            process.stack.store(self.next, context_term);
            return Action::Continue;
        }
        let (signed, endianness) = match self.ty {
            BsMatchSkipType::BigUnsigned => (false, Endianness::Big),
            BsMatchSkipType::BigSigned => (true, Endianness::Big),
            BsMatchSkipType::LittleUnsigned => (false, Endianness::Little),
            BsMatchSkipType::LittleSigned => (true, Endianness::Little),
            BsMatchSkipType::NativeUnsigned => (false, Endianness::Native),
            BsMatchSkipType::NativeSigned => (true, Endianness::Native),
        };
        assert!(
            bitsize <= 64,
            "unexpected match size for bs_match_skip intrinsic: {}",
            bitsize
        );
        if signed {
            match matcher.match_ap_number::<i64, 8>(bitsize, endianness) {
                Some(i) if i == value => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    match clone_context_to_heap(context, process) {
                        Ok(next) => {
                            process.stack.store(self.next, next);
                            Action::Continue
                        }
                        Err(_) => {
                            let extra = process.heap.heap_top() as usize - heap_top;
                            process.gc_needed = mem::size_of::<MatchContext>() + extra;
                            process.ip -= 1;
                            GC.dispatch(emulator, process)
                        }
                    }
                }
                _ => {
                    process.stack.store(self.is_err, OpaqueTerm::TRUE);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                }
            }
        } else {
            let expected = value as u64;
            match matcher.match_ap_number::<u64, 8>(bitsize, endianness) {
                Some(i) if i == expected => {
                    process.stack.store(self.is_err, OpaqueTerm::FALSE);
                    match clone_context_to_heap(context, process) {
                        Ok(next) => {
                            process.stack.store(self.next, next);
                            Action::Continue
                        }
                        Err(_) => {
                            let extra = process.heap.heap_top() as usize - heap_top;
                            process.gc_needed = mem::size_of::<MatchContext>() + extra;
                            process.ip -= 1;
                            GC.dispatch(emulator, process)
                        }
                    }
                }
                _ => {
                    process.stack.store(self.is_err, OpaqueTerm::TRUE);
                    process.stack.store(self.next, context_term);
                    Action::Continue
                }
            }
        }
    }
}
impl Inst for ops::BsTestTail {
    #[inline]
    fn dispatch(&self, _emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let context = process.stack.load(self.context);
        assert!(context.is_match_context());
        let context = unsafe { Gc::from_raw(context.as_ptr() as *mut MatchContext) };
        process
            .stack
            .store(self.dest, (context.bits_remaining() != self.size).into());
        Action::Continue
    }
}
impl Inst for ops::FuncInfo {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let fp = process.stack.frame_pointer();
        let sp = process.stack.stack_pointer();
        let current_frame_size = sp - fp - 2;
        let frame_size = self.frame_size as usize;
        // If this call requires a larger frame, allocate the needed slots
        //
        // Otherwise, if the call has a smaller frame than the one it is
        // replacing, truncate the stack pointer so that the extra slots
        // are no longer considered live
        //
        // We must also ensure that all stack slots which are allocated but
        // unused on entry, i.e. do not contain arguments to the function,
        // are zeroed.
        if frame_size > current_frame_size {
            unsafe {
                process.stack.alloca(frame_size - current_frame_size);
            }
        } else if frame_size < current_frame_size {
            unsafe {
                process.stack.dealloc(current_frame_size - frame_size);
            }
        }
        // Write NONE to all of the slots not occupied by arguments
        process.stack.zero(ARG0_REG + self.arity as Register);
        if log_enabled!(target: "process", log::Level::Trace) {
            let fun = emulator.code.function_by_id(self.id);
            let argv = process
                .stack
                .select_registers(ARG0_REG, self.arity as usize)
                .iter()
                .fold(String::new(), |mut acc, arg| {
                    use std::fmt::Write;
                    if acc.is_empty() {
                        write!(&mut acc, "{}", arg).unwrap();
                    } else {
                        write!(&mut acc, ", {}", arg).unwrap();
                    }
                    acc
                });
            trace!(target: "process", "entering function {} with arguments [{}]", fun.mfa().unwrap(), argv);
            trace!(target: "process", "fp = {}, sp = {}", fp, process.stack.stack_pointer());
        }
        Action::Continue
    }
}
impl Inst for ops::Identity {
    #[inline(always)]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let pid = process.pid();
        match Gc::new_in(pid, process) {
            Ok(pid) => {
                process.stack.store(self.dest, pid.into());
                Action::Continue
            }
            _ => {
                process.gc_needed = mem::size_of::<Pid>();
                process.ip -= 1;
                GC.dispatch(emulator, process)
            }
        }
    }
}
impl Inst for ops::Spawn2 {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let link = self.opts.contains(ops::SpawnOpts::LINK);
        let monitor = self.opts.contains(ops::SpawnOpts::MONITOR);

        let mut layout = LayoutBuilder::new();
        layout.build_pid();
        if monitor {
            layout.build_reference();
            layout.build_tuple(2);
        }
        let needed = layout.finish().size();
        let heap_available = process.heap.heap_available();
        if heap_available < needed {
            process.gc_needed = needed;
            process.ip -= 1;
            return GC.dispatch(emulator, process);
        }

        let fun_term = process.stack.load(self.fun);
        match fun_term.into() {
            Term::Closure(fun) => {
                let mut spawn_opts = SpawnOpts::default();
                spawn_opts.link = link;
                if monitor {
                    spawn_opts.monitor = Some(Default::default());
                }
                match emulator.spawn(process, fun.mfa(), &[fun_term], spawn_opts) {
                    (spawned, Some(spawn_ref)) => {
                        let pid = Gc::new_in(spawned.pid(), process).unwrap();
                        let tuple =
                            Tuple::from_slice(&[pid.into(), spawn_ref.into()], process).unwrap();
                        process.stack.store(self.dest, tuple.into());
                    }
                    (spawned, None) => {
                        let pid = Gc::new_in(spawned.pid(), process).unwrap();
                        process.stack.store(self.dest, pid.into());
                    }
                }
                Action::Continue
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = fun_term;
                process.exception_info.args = Some(fun_term);
                emulator.handle_error(process)
            }
        }
    }
}
impl Inst for ops::Spawn3 {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let mfa = emulator.code.function_by_id(self.fun).mfa().unwrap();
        let link = self.opts.contains(ops::SpawnOpts::LINK);
        let monitor = self.opts.contains(ops::SpawnOpts::MONITOR);

        let mut layout = LayoutBuilder::new();
        layout.build_pid();
        if monitor {
            layout.build_reference();
            layout.build_tuple(2);
        }
        let needed = layout.finish().size();
        let heap_available = process.heap.heap_available();
        if heap_available < needed {
            process.gc_needed = needed;
            process.ip -= 1;
            return GC.dispatch(emulator, process);
        }

        let arglist = process.stack.load(self.args);
        let mut argv = SmallVec::<[OpaqueTerm; 6]>::default();
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for result in cons.iter_raw() {
                    if result.is_err() {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    argv.push(unsafe { result.unwrap_unchecked() });
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                return emulator.handle_error(process);
            }
        }

        let mut spawn_opts = SpawnOpts::default();
        spawn_opts.link = link;
        if monitor {
            spawn_opts.monitor = Some(Default::default());
        }
        match emulator.spawn(process, (*mfa).into(), argv.as_slice(), spawn_opts) {
            (spawned, Some(spawn_ref)) => {
                let pid = Gc::new_in(spawned.pid(), process).unwrap();
                let tuple = Tuple::from_slice(&[pid.into(), spawn_ref.into()], process).unwrap();
                process.stack.store(self.dest, tuple.into());
            }
            (spawned, None) => {
                let pid = Gc::new_in(spawned.pid(), process).unwrap();
                process.stack.store(self.dest, pid.into());
            }
        }
        Action::Continue
    }
}
impl Inst for ops::Spawn3Indirect {
    #[inline]
    fn dispatch(&self, emulator: &Emulator, process: &mut ProcessLock) -> Action {
        let link = self.opts.contains(ops::SpawnOpts::LINK);
        let monitor = self.opts.contains(ops::SpawnOpts::MONITOR);

        let mut layout = LayoutBuilder::new();
        layout.build_pid();
        if monitor {
            layout.build_reference();
            layout.build_tuple(2);
        }
        let needed = layout.finish().size();
        let heap_available = process.heap.heap_available();
        if heap_available < needed {
            process.gc_needed = needed;
            process.ip -= 1;
            return GC.dispatch(emulator, process);
        }

        let module = process.stack.load(self.module);
        if !module.is_atom() {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = module;
            return emulator.handle_error(process);
        }
        let function = process.stack.load(self.function);
        if !function.is_atom() {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = function;
            return emulator.handle_error(process);
        }
        let arglist = process.stack.load(self.args);
        let mut argv = SmallVec::<[OpaqueTerm; 6]>::default();
        match arglist.into() {
            Term::Nil => (),
            Term::Cons(cons) => {
                for result in cons.iter_raw() {
                    if result.is_err() {
                        process.exception_info.flags = ExceptionFlags::ERROR;
                        process.exception_info.reason = atoms::Badarg.into();
                        process.exception_info.value = arglist;
                        process.exception_info.trace = None;
                        return emulator.handle_error(process);
                    }
                    argv.push(unsafe { result.unwrap_unchecked() });
                }
            }
            _ => {
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = arglist;
                return emulator.handle_error(process);
            }
        }

        let mfa = ModuleFunctionArity {
            module: module.as_atom(),
            function: function.as_atom(),
            arity: argv.len() as u8,
        };
        let mut spawn_opts = SpawnOpts::default();
        spawn_opts.link = link;
        if monitor {
            spawn_opts.monitor = Some(Default::default());
        }
        match emulator.spawn(process, mfa, argv.as_slice(), spawn_opts) {
            (spawned, Some(spawn_ref)) => {
                let pid = Gc::new_in(spawned.pid(), process).unwrap();
                let tuple = Tuple::from_slice(&[pid.into(), spawn_ref.into()], process).unwrap();
                process.stack.store(self.dest, tuple.into());
            }
            (spawned, None) => {
                let pid = Gc::new_in(spawned.pid(), process).unwrap();
                process.stack.store(self.dest, pid.into());
            }
        }
        Action::Continue
    }
}

fn clone_context_to_heap(
    context: MatchContext,
    process: &mut ProcessLock,
) -> Result<OpaqueTerm, AllocError> {
    let mut ptr = Gc::<MatchContext>::new_uninit_in(process)?;
    ptr.write(context);
    let cloned = unsafe { ptr.assume_init() };
    Ok(cloned.into())
}
