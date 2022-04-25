use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::process::current_process;

#[export_name = "__lumen_print_exception"]
pub extern "C-unwind" fn print_exception(
    kind: Term,
    reason: Term,
    trace: &mut Trace,
) -> *mut Trace {
    let source = None;
    trace
        .print(&current_process(), kind, reason, source)
        .unwrap();
    trace
}
