use core::ops::Deref;

use liblumen_alloc::gc::GcBox;
use liblumen_rt::backtrace::Trace;
use liblumen_rt::error::ErlangException;
use liblumen_rt::function::ErlangResult;
use liblumen_rt::term::{Atom, Closure, Cons, Map, Tuple};
use liblumen_rt::term::{OpaqueTerm, Term, TermType};

use crate::scheduler;

#[export_name = "__lumen_builtin_malloc"]
pub extern "C-unwind" fn malloc(kind: TermType, size: usize) -> *mut () {
    scheduler::with_current(|scheduler| {
        let arc_proc = scheduler.current_process();
        let proc = arc_proc.deref();
        match kind {
            TermType::Cons => Cons::new_in(proc).unwrap().as_ptr().cast(),
            TermType::Tuple => Tuple::new_in(size, proc).unwrap().as_ptr().cast(),
            TermType::Map => GcBox::into_raw(Map::new_in(proc).unwrap()).cast(),
            TermType::Closure => {
                GcBox::into_raw(unsafe { Closure::with_capacity_in(size, proc).unwrap() }).cast()
            }
            ty => panic!("unexpected malloc type: {:?}", ty),
        }
    })
}

#[export_name = "__lumen_build_stacktrace"]
pub extern "C-unwind" fn capture_trace() -> *mut Trace {
    let trace = Trace::capture();
    Trace::into_raw(trace)
}

#[export_name = "__lumen_builtin_raise/3"]
pub extern "C-unwind" fn raise(
    kind: OpaqueTerm,
    reason: OpaqueTerm,
    trace: *mut Trace,
) -> *mut ErlangException {
    debug_assert!(!trace.is_null());
    let trace = unsafe { Trace::from_raw(trace) };
    let kind: Term = kind.into();
    let kind: Atom = kind.try_into().unwrap();
    let exception = match kind.as_str() {
        "throw" | "error" | "exit" => ErlangException::new(kind, reason.into(), trace),
        other => panic!("invalid exception kind: {}", &other),
    };
    Box::into_raw(exception)
}

#[export_name = "__lumen_cleanup_exception"]
pub extern "C-unwind" fn cleanup(ptr: *mut ErlangException) {
    let _ = unsafe { Box::from_raw(ptr) };
}

#[export_name = "__lumen_builtin_yield"]
pub unsafe extern "C-unwind" fn process_yield() -> bool {
    scheduler::with_current(|scheduler| scheduler.process_yield())
}

#[export_name = "__lumen_builtin_exit"]
pub unsafe extern "C-unwind" fn process_exit(result: ErlangResult) {
    scheduler::with_current(|scheduler| {
        match result {
            Ok(_) => {
                scheduler.current_process().exit_normal();
            }
            Err(err) => {
                scheduler.current_process().exit_error(err);
            }
        }

        scheduler.process_yield()
    });
}
