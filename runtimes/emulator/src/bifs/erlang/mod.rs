use std::io::Write;
use std::mem;
use std::ops::Deref;
use std::sync::atomic::Ordering;

use firefly_alloc::heap::Heap;
use firefly_number::Int;
use firefly_rt::error::ExceptionFlags;
use firefly_rt::function::{ErlangResult, ModuleFunctionArity};
use firefly_rt::gc::{garbage_collect, Gc, RootSet};
use firefly_rt::process::monitor::{Monitor, MonitorEntry, MonitorFlags, UnaliasMode};
use firefly_rt::process::signals::Signal;
use firefly_rt::process::{ProcessFlags, ProcessLock, StatusFlags, ARG0_REG};
use firefly_rt::scheduler::Scheduler;
use firefly_rt::services::registry::{self, WeakAddress};
use firefly_rt::term::*;

use log::warn;

use smallvec::SmallVec;

use crate::emulator::{current_scheduler, Action};
use crate::{badarg, unwrap_or_badarg};

macro_rules! handle_arith_result {
    ($process:expr, $term:expr, $math:expr) => {
        match $math {
            Ok(Number::Float(n)) => ErlangResult::Ok(n.into()),
            Ok(Number::Integer(n)) => handle_safe_integer_arith_result!($process, n),
            Err(_) => badarg!($process, $term),
        }
    };
}

macro_rules! handle_integer_arith_result {
    ($process:expr, $term:expr, $math:expr) => {
        match $math {
            Ok(result) => handle_safe_integer_arith_result!($process, result),
            Err(_) => badarg!($process, $term),
        }
    };
}

macro_rules! handle_safe_integer_arith_result {
    ($process:expr, $math:expr) => {
        match $math {
            Int::Small(i) => ErlangResult::Ok(i.try_into().unwrap()),
            Int::Big(i) => {
                let p = $process;
                let i = BigInt::from(i);
                match Gc::new_uninit_in(p) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        ErlangResult::Ok(empty.assume_init().into())
                    },
                    Err(_) => {
                        assert!(garbage_collect(p, Default::default()).is_ok());
                        let boxed = Gc::new_in(i, p).unwrap();
                        ErlangResult::Ok(boxed.into())
                    }
                }
            }
        }
    };
}

#[export_name = "erlang:+/2"]
pub extern "C-unwind" fn plus2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l + r)
}

#[export_name = "erlang:-/1"]
pub extern "C-unwind" fn neg1(process: &mut ProcessLock, lhs: OpaqueTerm) -> ErlangResult {
    let l: Term = lhs.into();
    handle_arith_result!(process, lhs, -l)
}

#[export_name = "erlang:-/2"]
pub extern "C-unwind" fn minus2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l - r)
}

#[export_name = "erlang:*/2"]
pub extern "C-unwind" fn mul2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    handle_arith_result!(process, lhs, l * r)
}

#[export_name = "erlang://2"]
pub extern "C-unwind" fn divide2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    match l / r {
        Ok(result) => handle_arith_result!(process, lhs, result),
        Err(_) => badarg!(process, rhs),
    }
}

#[export_name = "erlang:div/2"]
pub extern "C-unwind" fn div2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let l: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let r: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_integer_arith_result!(process, rhs, l / r)
}

#[export_name = "erlang:rem/2"]
pub extern "C-unwind" fn rem2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let l: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let r: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_integer_arith_result!(process, rhs, l % r)
}

#[export_name = "erlang:bsl/2"]
pub extern "C-unwind" fn bsl2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs << rhs)
}

#[export_name = "erlang:bsr/2"]
pub extern "C-unwind" fn bsr2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs >> rhs)
}

#[export_name = "erlang:band/2"]
pub extern "C-unwind" fn band2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs & rhs)
}

#[export_name = "erlang:bnot/1"]
pub extern "C-unwind" fn bnot1(process: &mut ProcessLock, lhs: OpaqueTerm) -> ErlangResult {
    let l: Term = lhs.into();
    match l {
        Term::Bool(b) => ErlangResult::Ok((!b).into()),
        Term::Int(i) => {
            let i = !i;
            let term: Result<OpaqueTerm, _> = i.try_into();
            match term {
                Ok(t) => ErlangResult::Ok(t),
                Err(_) => loop {
                    match Gc::new_uninit_in(process) {
                        Ok(mut empty) => unsafe {
                            empty.write(BigInt::from(i));
                            return ErlangResult::Ok(empty.assume_init().into());
                        },
                        Err(_) => {
                            assert!(garbage_collect(process, Default::default()).is_ok())
                        }
                    }
                },
            }
        }
        Term::BigInt(i) => {
            let i = !(i.as_ref());
            loop {
                match Gc::new_uninit_in(process) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        return ErlangResult::Ok(empty.assume_init().into());
                    },
                    Err(_) => {
                        assert!(garbage_collect(process, Default::default()).is_ok())
                    }
                }
            }
        }
        _ => badarg!(process, lhs),
    }
}

#[export_name = "erlang:bor/2"]
pub extern "C-unwind" fn bor2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs | rhs)
}

#[export_name = "erlang:bxor/2"]
pub extern "C-unwind" fn bxor2(
    process: &mut ProcessLock,
    lhs: OpaqueTerm,
    rhs: OpaqueTerm,
) -> ErlangResult {
    let l: Term = lhs.into();
    let r: Term = rhs.into();
    let lhs: Int = unwrap_or_badarg!(process, lhs, l.try_into());
    let rhs: Int = unwrap_or_badarg!(process, rhs, r.try_into());

    handle_safe_integer_arith_result!(process, lhs ^ rhs)
}

#[export_name = "erlang:abs/1"]
pub extern "C-unwind" fn abs(process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    match term.into() {
        Term::Int(i) => {
            let i = i.abs();
            if i > Int::MAX_SMALL {
                let i = BigInt::from(i);
                loop {
                    match Gc::new_uninit_in(process) {
                        Ok(mut empty) => unsafe {
                            empty.write(i);
                            return ErlangResult::Ok(empty.assume_init().into());
                        },
                        Err(_) => {
                            assert!(garbage_collect(process, Default::default()).is_ok());
                        }
                    }
                }
            } else {
                ErlangResult::Ok(Term::Int(i).into())
            }
        }
        Term::BigInt(i) => {
            let i = i.abs();
            loop {
                match Gc::new_uninit_in(process) {
                    Ok(mut empty) => unsafe {
                        empty.write(i);
                        return ErlangResult::Ok(empty.assume_init().into());
                    },
                    Err(_) => {
                        assert!(garbage_collect(process, Default::default()).is_ok());
                    }
                }
            }
        }
        Term::Float(f) => ErlangResult::Ok(Term::Float(f.abs()).into()),
        _ => badarg!(process, term),
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

#[export_name = "erlang:display/1"]
pub extern "C-unwind" fn display(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:debug/1"]
pub extern "C-unwind" fn debug(_process: &mut ProcessLock, term: OpaqueTerm) -> ErlangResult {
    let term: Term = term.into();
    println!("{:?}", &term);
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_nl/0"]
pub extern "C-unwind" fn display_nl(_process: &mut ProcessLock) -> ErlangResult {
    println!();
    ErlangResult::Ok(true.into())
}

#[export_name = "erlang:display_string/1"]
pub extern "C-unwind" fn display_string(
    process: &mut ProcessLock,
    term: OpaqueTerm,
) -> ErlangResult {
    let list: Term = term.into();
    match list {
        Term::Nil => ErlangResult::Ok(true.into()),
        Term::Cons(cons) => {
            match cons.as_ref().to_string() {
                Some(ref s) => print!("{}", s),
                None => badarg!(process, term),
            }
            ErlangResult::Ok(true.into())
        }
        _other => badarg!(process, term),
    }
}

#[export_name = "erlang:puts/1"]
pub extern "C-unwind" fn puts(_process: &mut ProcessLock, printable: OpaqueTerm) -> ErlangResult {
    let printable: Term = printable.into();

    let bits = printable.as_bitstring().unwrap();
    assert!(bits.is_aligned());
    assert!(bits.is_binary());
    let bytes = unsafe { bits.as_bytes_unchecked() };
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes).unwrap();
    ErlangResult::Ok(true.into())
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
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = id;
            process.exception_info.trace = None;
            ErlangResult::Err
        }
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
                process.exception_info.flags = ExceptionFlags::ERROR;
                process.exception_info.reason = atoms::Badarg.into();
                process.exception_info.value = opts;
                process.exception_info.trace = None;
                return ErlangResult::Err;
            }

            MonitorFlags::empty() | mode
        }
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = opts;
            process.exception_info.trace = None;
            return ErlangResult::Err;
        }
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

    process.exception_info.flags = ExceptionFlags::ERROR;
    process.exception_info.reason = atoms::Badarg.into();
    process.exception_info.value = alias;
    process.exception_info.trace = None;
    ErlangResult::Err
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
        _ => {
            process.exception_info.flags = ExceptionFlags::ERROR;
            process.exception_info.reason = atoms::Badarg.into();
            process.exception_info.value = pid_term;
            process.exception_info.trace = None;
            ErlangResult::Err
        }
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
