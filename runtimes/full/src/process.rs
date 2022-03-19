pub mod out_of_code;

use std::ffi::c_void;
use std::mem::transmute;

use liblumen_core::sys::dynamic_call::DynamicCallee;

use liblumen_alloc::erts::process::ffi::ProcessSignal;
use liblumen_alloc::erts::process::{Frame, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

pub use lumen_rt_core::process::{current_process, monitor, replace_log_exit, set_log_exit, spawn};

#[no_mangle]
pub unsafe extern "C-unwind" fn __lumen_panic(term: Term) {
    panic!("{}", term);
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
    callee: DynamicCallee,
    arguments: Vec<Term>,
) -> Term {
    let native = unsafe {
        let ptr = transmute::<DynamicCallee, *const c_void>(callee);

        Native::from_ptr(ptr, arguments.len() as Arity)
    };

    let frame = Frame::new(module_function_arity, native);
    let frame_with_arguments = frame.with_arguments(false, &arguments);
    current_process().queue_frame_with_arguments(frame_with_arguments);

    Term::NONE
}

#[export_name = "__lumen_process_signal"]
#[thread_local]
static mut PROCESS_SIGNAL: ProcessSignal = ProcessSignal::None;
