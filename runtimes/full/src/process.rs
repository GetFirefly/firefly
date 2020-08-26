pub mod out_of_code;

use liblumen_alloc::erts::process::ffi::{process_error, ProcessSignal};
use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

pub use lumen_rt_core::process::{current_process, monitor, replace_log_exit, set_log_exit, spawn};

#[unwind(allowed)]
#[no_mangle]
pub unsafe extern "C" fn __lumen_start_panic(_payload: usize) {
    panic!(process_error().unwrap());
}

#[export_name = "lumen_rt_apply_2"]
pub fn apply_2(function_boxed_closure: Boxed<Closure>, arguments: Vec<Term>) -> Term {
    let frame_with_arguments = function_boxed_closure.frame_with_arguments(false, arguments);
    current_process().queue_frame_with_arguments(frame_with_arguments);

    Term::NONE
}

#[export_name = "lumen_rt_apply_3"]
pub fn apply_3(
    module_function_arity: ModuleFunctionArity,
    native: Native,
    arguments: Vec<Term>,
) -> Term {
    let frame = Frame::new(module_function_arity, native);
    let frame_with_arguments = frame.with_arguments(false, &arguments);
    current_process().queue_frame_with_arguments(frame_with_arguments);

    Term::NONE
}

#[export_name = "__lumen_process_signal"]
#[thread_local]
static mut PROCESS_SIGNAL: ProcessSignal = ProcessSignal::None;
