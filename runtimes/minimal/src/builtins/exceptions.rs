use liblumen_alloc::erts::process::ffi::process_raise;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::process::current_process;

#[export_name = "__lumen_builtin_fail/1"]
pub extern "C" fn builtin_fail(reason: Term) -> Term {
    use liblumen_alloc::erts::exception::{self, ArcError, RuntimeException};
    if reason.is_none() {
        reason
    } else {
        let trace = Trace::capture();
        let err = RuntimeException::Error(exception::Error::new_with_trace(reason, None, trace));
        process_raise(&current_process(), err);
    }
}

#[export_name = "__lumen_builtin_trace.capture"]
pub extern "C" fn builtin_trace_capture() -> *mut Trace {
    let trace = Trace::capture();
    Trace::into_raw(trace)
}

#[export_name = "__lumen_builtin_trace.print"]
pub extern "C" fn builtin_trace_print(kind: Term, reason: Term, trace: &mut Trace) -> *mut Trace {
    trace.print(kind, reason);
    trace
}

#[export_name = "__lumen_builtin_trace.construct"]
pub extern "C" fn builtin_trace_construct(trace: &mut Trace) -> Term {
    trace.as_term().unwrap()
}
