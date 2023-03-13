mod debugging;
mod operators;
mod signals;

pub use self::debugging::*;
pub use self::operators::*;
pub use self::signals::*;

use std::cmp;
use std::sync::atomic::Ordering;

use firefly_alloc::heap::Heap;
use firefly_rt::error::ExceptionFlags;
use firefly_rt::function::{self, ErlangResult, ModuleFunctionArity};
use firefly_rt::gc::{garbage_collect, Gc, RootSet};
use firefly_rt::process::{Priority, Process, ProcessFlags, ProcessLock, StatusFlags};
use firefly_rt::scheduler::Scheduler;
use firefly_rt::term::*;

use log::warn;

use smallvec::SmallVec;

use crate::badarg;
use crate::emulator::current_scheduler;

#[export_name = "erlang:is_builtin/3"]
pub extern "C-unwind" fn is_builtin3(
    process: &mut ProcessLock,
    m: OpaqueTerm,
    f: OpaqueTerm,
    a: OpaqueTerm,
) -> ErlangResult {
    if !m.is_atom() {
        badarg!(process, m);
    }
    if !f.is_atom() {
        badarg!(process, f);
    }

    match a.into() {
        Term::Int(i) if i >= 0 && i < 256 => {
            // If a native function symbol is defined for this MFA, it is a builtin
            let mfa = ModuleFunctionArity {
                module: m.as_atom(),
                function: f.as_atom(),
                arity: i as u8,
            };
            ErlangResult::Ok(function::find_symbol(&mfa).is_some().into())
        }
        _ => badarg!(process, a),
    }
}

#[export_name = "erlang:make_ref/0"]
pub extern "C-unwind" fn make_ref0(process: &mut ProcessLock) -> ErlangResult {
    let ref_id = current_scheduler().next_reference_id();
    loop {
        match Gc::new_uninit_in(process) {
            Ok(mut empty) => unsafe {
                empty.write(Reference::new(ref_id));
                return ErlangResult::Ok(empty.assume_init().into());
            },
            Err(_) => {
                assert!(garbage_collect(process, Default::default()).is_ok());
            }
        }
    }
}

#[export_name = "erlang:list_to_atom/1"]
pub extern "C-unwind" fn list_to_atom(process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    match term.into() {
        Term::Nil => return ErlangResult::Ok(atoms::Empty.into()),
        Term::Cons(cons) => {
            if let Some(s) = cons.as_ref().to_string() {
                return ErlangResult::Ok(Atom::str_to_term(&s));
            }
        }
        _ => (),
    }
    badarg!(process, term)
}

#[export_name = "erlang:binary_to_list/1"]
pub extern "C-unwind" fn binary_to_list(
    process: &mut ProcessLock,
    term: OpaqueTerm,
) -> ErlangResult {
    let t: Term = term.into();
    if let Some(bits) = t.as_bitstring() {
        assert!(bits.is_binary());
        assert!(bits.is_aligned());

        let bytes = unsafe { bits.as_bytes_unchecked() };
        let result = match core::str::from_utf8(bytes).ok() {
            Some(s) => Cons::charlist_from_str(s, process).unwrap(),
            None => Cons::from_bytes(bytes, process).unwrap(),
        };
        match result {
            None => ErlangResult::Ok(Term::Nil.into()),
            Some(cons) => ErlangResult::Ok(cons.into()),
        }
    } else {
        badarg!(process, term)
    }
}

#[export_name = "erlang:spawn/2"]
pub extern "C-unwind" fn spawn2(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _fun: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn/2 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn/4"]
pub extern "C-unwind" fn spawn4(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _module: OpaqueTerm,
    _function: OpaqueTerm,
    _args: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn/4 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_link/2"]
pub extern "C-unwind" fn spawn_link2(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _fun: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_link/2 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_link/4"]
pub extern "C-unwind" fn spawn_link4(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _module: OpaqueTerm,
    _function: OpaqueTerm,
    _args: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_link/4 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_monitor/2"]
pub extern "C-unwind" fn spawn_monitor2(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _fun: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_monitor/2 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_monitor/4"]
pub extern "C-unwind" fn spawn_monitor4(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _module: OpaqueTerm,
    _function: OpaqueTerm,
    _args: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_monitor/4 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_opt/2"]
pub extern "C-unwind" fn spawn_opt2(
    process: &mut ProcessLock,
    mut fun_term: OpaqueTerm,
    opts: OpaqueTerm,
) -> ErlangResult {
    use firefly_rt::process::SpawnOpts;

    let opts: Term = opts.into();
    let mut spawn_opts: SpawnOpts = match opts.try_into() {
        Ok(opts) => opts,
        Err(_) => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = fun_term;
            process.exception_info.trace = None;
            return ErlangResult::Err;
        }
    };
    let monitor = spawn_opts.monitor.is_some();

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
        let mut roots = RootSet::default();
        if let Some(monitor_opts) = spawn_opts.monitor.as_mut() {
            roots += &mut monitor_opts.tag as *mut _;
        }
        roots += &mut fun_term as *mut _;
        assert!(garbage_collect(process, roots).is_ok());
    }

    match fun_term.into() {
        Term::Closure(fun) => {
            match current_scheduler().spawn(process, fun.mfa(), &[fun_term], spawn_opts) {
                (spawned, Some(spawn_ref)) => {
                    let pid = Gc::new_in(spawned.pid(), process).unwrap();
                    let tuple =
                        Tuple::from_slice(&[pid.into(), spawn_ref.into()], process).unwrap();
                    ErlangResult::Ok(tuple.into())
                }
                (spawned, None) => {
                    let pid = Gc::new_in(spawned.pid(), process).unwrap();
                    ErlangResult::Ok(pid.into())
                }
            }
        }
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = fun_term;
            process.exception_info.trace = None;
            ErlangResult::Err
        }
    }
}

#[export_name = "erlang:spawn_opt/3"]
pub extern "C-unwind" fn spawn_opt3(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _fun: OpaqueTerm,
    _opts: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_opt/4 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:spawn_opt/4"]
pub extern "C-unwind" fn spawn_opt4(
    process: &mut ProcessLock,
    module: OpaqueTerm,
    function: OpaqueTerm,
    mut args: OpaqueTerm,
    opts: OpaqueTerm,
) -> ErlangResult {
    use firefly_rt::process::SpawnOpts;

    if !module.is_atom() {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = module;
        return ErlangResult::Err;
    }
    if !function.is_atom() {
        process.exception_info.flags = ExceptionFlags::ERROR;
        process.exception_info.reason = atoms::Badarg.into();
        process.exception_info.value = function;
        return ErlangResult::Err;
    }

    let opts_term: Term = opts.into();
    let mut spawn_opts: SpawnOpts = match opts_term.try_into() {
        Ok(opts) => opts,
        Err(_) => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = opts;
            process.exception_info.trace = None;
            return ErlangResult::Err;
        }
    };
    let monitor = spawn_opts.monitor.is_some();

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
        let mut roots = RootSet::default();
        if let Some(monitor_opts) = spawn_opts.monitor.as_mut() {
            roots += &mut monitor_opts.tag as *mut _;
        }
        roots += &mut args as *mut _;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let mut arity = 0;
    let mut argv = SmallVec::<[OpaqueTerm; 4]>::default();
    match args.into() {
        Term::Nil => (),
        Term::Cons(cons) => {
            for result in cons.iter_raw() {
                if result.is_err() {
                    process.exception_info.flags = ExceptionFlags::ERROR;
                    process.exception_info.reason = atoms::Badarg.into();
                    process.exception_info.value = args;
                    process.exception_info.trace = None;
                    return ErlangResult::Err;
                }
                arity += 1;
                argv.push(unsafe { result.unwrap_unchecked() });
            }
        }
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = args;
            process.exception_info.trace = None;
            return ErlangResult::Err;
        }
    }

    let mfa = ModuleFunctionArity {
        module: module.as_atom(),
        function: function.as_atom(),
        arity,
    };

    match current_scheduler().spawn(process, mfa, argv.as_slice(), spawn_opts) {
        (spawned, Some(spawn_ref)) => {
            let pid = Gc::new_in(spawned.pid(), process).unwrap();
            let tuple = Tuple::from_slice(&[pid.into(), spawn_ref.into()], process).unwrap();
            ErlangResult::Ok(tuple.into())
        }
        (spawned, None) => {
            let pid = Gc::new_in(spawned.pid(), process).unwrap();
            ErlangResult::Ok(pid.into())
        }
    }
}

#[export_name = "erlang:spawn_opt/5"]
pub extern "C-unwind" fn spawn_opt5(
    process: &mut ProcessLock,
    node: OpaqueTerm,
    _module: OpaqueTerm,
    _function: OpaqueTerm,
    _args: OpaqueTerm,
    _opts: OpaqueTerm,
) -> ErlangResult {
    warn!(target: "process", "spawn_opt/5 is unavailable, distribution is unimplemented");
    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Notsup.into();
    process.exception_info.value = node;
    process.exception_info.trace = None;
    ErlangResult::Err
}

#[export_name = "erlang:process_flag/2"]
pub extern "C-unwind" fn process_flag2(
    process: &mut ProcessLock,
    flag: OpaqueTerm,
    value: OpaqueTerm,
) -> ErlangResult {
    use firefly_rt::process::MaxHeapSize;

    if !flag.is_atom() {
        badarg!(process, flag);
    }
    let flag_atom = flag.as_atom();
    match flag_atom.as_str() {
        "trap_exit" => {
            let prev_value = process.flags.contains(ProcessFlags::TRAP_EXIT);
            match value {
                OpaqueTerm::TRUE => {
                    process.flags |= ProcessFlags::TRAP_EXIT;
                    ErlangResult::Ok(prev_value.into())
                }
                OpaqueTerm::FALSE => {
                    process.flags.remove(ProcessFlags::TRAP_EXIT);
                    ErlangResult::Ok(prev_value.into())
                }
                _ => badarg!(process, value),
            }
        }
        "error_handler" => match value.into() {
            Term::Atom(handler) => {
                let prev = process
                    .as_ref()
                    .error_handler
                    .swap(handler, Ordering::Relaxed);
                ErlangResult::Ok(prev.into())
            }
            _ => badarg!(process, value),
        },
        "fullsweep_after" => match value.into() {
            Term::Int(i) if i >= 0 => match usize::try_from(i) {
                Ok(fullsweep_after) => {
                    let prev = process
                        .as_ref()
                        .fullsweep_after
                        .swap(fullsweep_after, Ordering::Relaxed);
                    ErlangResult::Ok(Term::Int(prev.try_into().unwrap()).into())
                }
                Err(_) => badarg!(process, value),
            },
            _ => badarg!(process, value),
        },
        "max_heap_size" => {
            let max_heap_size: Term = value.into();
            match MaxHeapSize::try_from(max_heap_size) {
                Ok(max_heap_size) => {
                    let prev = process
                        .as_ref()
                        .max_heap_size
                        .swap(max_heap_size, Ordering::Relaxed);
                    // If no limit is set, use the integer representation
                    if prev.size.is_none() {
                        return ErlangResult::Ok(Term::Int(0).into());
                    }
                    let prev_size: Term = prev.size.map(|sz| sz.get()).unwrap().try_into().unwrap();
                    // If the system defaults are being used, use the integer representation
                    if prev.kill && prev.error_logger {
                        return ErlangResult::Ok(prev_size.into());
                    }
                    // Otherwise allocate a map
                    let mut map = match Map::with_capacity_in(3, process) {
                        Ok(map) => map,
                        Err(_) => {
                            assert!(garbage_collect(process, Default::default()).is_ok());
                            Map::with_capacity_in(3, process).unwrap()
                        }
                    };
                    map.put_mut(atoms::ErrorLogger, prev.error_logger);
                    map.put_mut(atoms::Kill, prev.kill);
                    map.put_mut(atoms::Size, prev_size);
                    ErlangResult::Ok(map.into())
                }
                Err(_) => badarg!(process, value),
            }
        }
        "message_queue_data" => {
            // This flag has no effect in our implementation, we simply return the current value
            if !value.is_atom() {
                badarg!(process, value);
            }
            ErlangResult::Ok(atoms::OffHeap.into())
        }
        "priority" => {
            let prio: Term = value.into();
            match Priority::try_from(prio) {
                Ok(prio) => {
                    let mut status = process.status(Ordering::Acquire);
                    let prev = status.priority();
                    loop {
                        let new_status = status | prio;
                        match process.cmpxchg_status_flags(status, new_status) {
                            Ok(_) => break,
                            Err(current) => {
                                status = current;
                            }
                        }
                    }
                    ErlangResult::Ok(prev.into())
                }
                Err(_) => badarg!(process, value),
            }
        }
        "min_heap_size" => {
            // This flag cannot be set after spawn currently, return the current value
            if let Term::Int(_) = value.into() {
                let prev = process
                    .as_ref()
                    .min_heap_size
                    .map(|sz| sz.get())
                    .unwrap_or(0);
                let prev: Term = prev.try_into().unwrap();
                ErlangResult::Ok(prev.into())
            } else {
                badarg!(process, value)
            }
        }
        "min_bin_vheap_size" => {
            // This flag cannot be set after spawn currently, return the current value
            if let Term::Int(_) = value.into() {
                let prev = process
                    .as_ref()
                    .min_bin_vheap_size
                    .map(|sz| sz.get())
                    .unwrap_or(0);
                let prev: Term = prev.try_into().unwrap();
                ErlangResult::Ok(prev.into())
            } else {
                badarg!(process, value)
            }
        }
        "save_calls" => {
            // This flag has no effect currently, always return 0
            if let Term::Int(_) = value.into() {
                ErlangResult::Ok(Term::Int(0).into())
            } else {
                badarg!(process, value)
            }
        }
        "sensitive" => {
            let is_sensitive = match value {
                OpaqueTerm::TRUE => true,
                OpaqueTerm::FALSE => false,
                _ => badarg!(process, value),
            };
            let mut status = process.status(Ordering::Acquire);
            let prev = status.contains(StatusFlags::SENSITIVE);
            if prev == is_sensitive {
                return ErlangResult::Ok(prev.into());
            }
            loop {
                let new_status = if is_sensitive {
                    status | StatusFlags::SENSITIVE
                } else {
                    status & !StatusFlags::SENSITIVE
                };
                match process.cmpxchg_status_flags(status, new_status) {
                    Ok(_) => break,
                    Err(current) => {
                        status = current;
                    }
                }
            }
            ErlangResult::Ok(prev.into())
        }
        _ => badarg!(process, flag),
    }
}

#[export_name = "erlang:process_flag/3"]
pub extern "C-unwind" fn process_flag3(
    process: &mut ProcessLock,
    pid: OpaqueTerm,
    flag: OpaqueTerm,
    value: OpaqueTerm,
) -> ErlangResult {
    if let Term::Pid(_) = pid.into() {
        if !flag.is_atom() {
            badarg!(process, flag);
        }
        let flag_atom = flag.as_atom();
        match flag_atom.as_str() {
            "save_calls" => {
                // This flag has no effect currently, always return 0
                if let Term::Int(_) = value.into() {
                    ErlangResult::Ok(Term::Int(0).into())
                } else {
                    badarg!(process, value)
                }
            }
            _ => badarg!(process, value),
        }
    } else {
        badarg!(process, pid)
    }
}

#[export_name = "erlang:yield/0"]
pub extern "C-unwind" fn yield0(process: &mut ProcessLock) -> ErlangResult {
    // Force a yield when this function returns
    //
    // This is slightly different than the behavior of the Yield instruction, which
    // yields before it conceptually returns, but there is no practical difference in
    // behavior.
    process.reductions = Process::MAX_REDUCTIONS;
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:bump_reductions/1"]
pub extern "C-unwind" fn bump_reductions(
    process: &mut ProcessLock,
    reds: OpaqueTerm,
) -> ErlangResult {
    match reds.into() {
        Term::Int(i) if i >= 1 => {
            if let Ok(i) = usize::try_from(i) {
                // Clamp large values to the maximum number of reductions
                process.reductions = cmp::min(
                    Process::MAX_REDUCTIONS,
                    process.reductions.saturating_add(i),
                );
                return ErlangResult::Ok(true.into());
            }
        }
        _ => (),
    }

    badarg!(process, reds);
}
