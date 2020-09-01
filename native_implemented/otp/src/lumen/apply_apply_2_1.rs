//! This is used as the `init_fn` for `Scheduler::spawn_closure`, as the spawning code can only
//! pass at most 1 argument and `erlang:apply/2` takes two arguments

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badarity};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::apply::arguments_term_to_vec;

#[native_implemented::function(lumen:apply_apply_2/1)]
fn result(process: &Process, arguments: Term) -> exception::Result<Term> {
    let argument_vec = arguments_term_to_vec(arguments)?;
    let arguments_len = argument_vec.len();

    if arguments_len == (erlang::apply_2::ARITY as usize) {
        let apply_2_function = argument_vec[0];
        let apply_2_arguments = argument_vec[1];

        // want to call into `erlang:apply/2` `native` and not `result` so that the stack trace
        // shows `erlang:apply/2`.
        Ok(erlang::apply_2::native(apply_2_function, apply_2_arguments))
    } else {
        let function = process.export_closure(
            erlang::module(),
            erlang::apply_2::function(),
            erlang::apply_2::ARITY,
            erlang::apply_2::CLOSURE_NATIVE,
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
                    erlang::apply_2::ARITY
                )
                .into(),
            ),
        ))
    }
}
