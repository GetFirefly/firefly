use anyhow::*;

use liblumen_core::sys::dynamic_call::DynamicCallee;

use liblumen_alloc::erts::apply::find_symbol;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

use crate::erlang::apply::arguments_term_to_vec;

extern "Rust" {
    #[link_name = "lumen_rt_apply_3"]
    fn runtime_apply_3(
        module_function_arity: ModuleFunctionArity,
        callee: DynamicCallee,
        arguments: Vec<Term>,
    ) -> Term;
}

#[native_implemented::function(erlang:apply/3)]
fn result(module: Term, function: Term, arguments: Term) -> exception::Result<Term> {
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
                        module_atom.name(),
                        function_atom.name(),
                        arity
                    )
                    .into(),
                ),
            )
            .into())
        }
    }
}
