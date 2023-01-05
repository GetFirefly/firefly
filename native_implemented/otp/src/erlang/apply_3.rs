use std::ptr::NonNull;

use anyhow::*;
use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;

use firefly_rt::function::{Arity, DynamicCallee, ErlangResult, find_symbol, ModuleFunctionArity};
use firefly_rt::term::{atoms, Term};

use crate::erlang::apply::arguments_term_to_vec;

extern "Rust" {
    #[link_name = "lumen_rt_apply_3"]
    fn runtime_apply_3(
        module_function_arity: ModuleFunctionArity,
        callee: DynamicCallee,
        arguments: Vec<Term>,
    ) -> ErlangResult;
}

#[export_name = "erlang:apply/3"]
pub extern "C-unwind" fn apply_3(module: Term, function: Term, arguments: Term) -> ErlangResult {
    let arc_process = crate::runtime::process::current_process();
    match apply_3_impl(module, function, arguments) {
        Ok(result) => result,
        Err(exception) => arc_process.return_status(Err(exception)),
    }
}
fn apply_3_impl(module: Term, function: Term, arguments: Term) -> Result<ErlangResult, NonNull<ErlangException>> {
    let module_atom = term_try_into_atom!(module)?;
    let function_atom = term_try_into_atom!(function)?;
    let argument_vec = arguments_term_to_vec(arguments)?;
    let arity = argument_vec.len() as Arity;

    let module_function_arity = ModuleFunctionArity {
        module: module_atom,
        function: function_atom,
        arity,
    };

    match find_symbol(&module_function_arity) {
        Some(callee) => Ok(unsafe { runtime_apply_3(module_function_arity, callee, argument_vec) }),
        None => {
            let trace = Trace::capture();
            trace.set_top_frame(&module_function_arity, argument_vec.as_slice());
            Err(exception::undef(
                trace,
                Some(
                    anyhow!(
                        "{}:{}/{} is not exported",
                        module_atom.as_str(),
                        function_atom.as_str(),
                        arity
                    )
                    .into(),
                ),
            )
            .into())
        }
    }
}

pub fn frame() -> Frame {
    frame_for_native(NATIVE)
}

pub fn frame_for_native(native: Native) -> Frame {
    Frame::new(module_function_arity(), native)
}

pub fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: atoms::Erlang,
        function: atoms::Apply,
        arity: ARITY,
    }
}

pub const ARITY: Arity = 3;
pub const CLOSURE_NATIVE: Option<NonNull<std::ffi::c_void>> =
    Some(unsafe { NonNull::new_unchecked(apply_3 as *mut std::ffi::c_void) });
