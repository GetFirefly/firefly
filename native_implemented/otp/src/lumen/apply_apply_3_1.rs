//! This is used as the `init_fn` for `Scheduler::spawn_module_function_arguments`, as the spawning
//! code can only pass at most 1 argument and `erlang:apply/3` takes three arguments

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badarity};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::apply::arguments_term_to_vec;

#[native_implemented::function(lumen:apply_apply_3/1)]
fn result(process: &Process, arguments: Term) -> exception::Result<Term> {
    let argument_vec = arguments_term_to_vec(arguments)?;
    let arguments_len = argument_vec.len();

    if arguments_len == (erlang::apply_3::ARITY as usize) {
        let apply_3_module = argument_vec[0];
        let apply_3_function = argument_vec[1];
        let apply_3_arguments = argument_vec[2];

        // want to call into `erlang:apply/3` `native` and not `result` so that the stack trace
        // shows `erlang:apply/3`.
        Ok(erlang::apply_3::native(
            apply_3_module,
            apply_3_function,
            apply_3_arguments,
        ))
    } else {
        let function = process.export_closure(
            erlang::module(),
            erlang::apply_3::function(),
            erlang::apply_3::ARITY,
            erlang::apply_3::CLOSURE_NATIVE,
        );

        Err(badarity(
            process,
            function,
            arguments,
            Trace::capture(),
            Some(
                anyhow!(
                    "function arguments {} is {} term(s), but should be {}",
                    arguments,
                    arguments_len,
                    erlang::apply_3::ARITY
                )
                .into(),
            ),
        ))
    }
}
