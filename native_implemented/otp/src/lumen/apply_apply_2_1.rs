//! This is used as the `init_fn` for `Scheduler::spawn_closure`, as the spawning code can only
//! pass at most 1 argument and `erlang:apply/2` takes two arguments
use anyhow::anyhow;

use liblumen_alloc::erts::exception::{badarity, Exception};
use liblumen_alloc::erts::process::ffi::ErlangResult;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;
use crate::erlang::apply::arguments_term_to_vec;

#[export_name = "lumen:apply_apply_2/1"]
pub extern "C-unwind" fn apply_apply_2(arguments: Term) -> ErlangResult {
    let arc_process = crate::runtime::process::current_process();
    let argument_vec = match arguments_term_to_vec(arguments) {
        Ok(args) => args,
        Err(err) => match err {
            Exception::Runtime(exception) => {
                return ErlangResult::error(arc_process.raise(exception));
            }
            Exception::System(ref exception) => {
                panic!("{}", exception)
            }
        },
    };
    let arguments_len = argument_vec.len();

    if arguments_len == 2 {
        let apply_2_function = argument_vec[0];
        let apply_2_arguments = argument_vec[1];

        // want to call into `erlang:apply/2` `native` and not `result` so that the stack trace
        // shows `erlang:apply/2`.
        erlang::apply_2::apply_2(apply_2_function, apply_2_arguments)
    } else {
        let function = arc_process.export_closure(
            erlang::module(),
            Atom::from_str("apply"),
            2,
            erlang::apply_2::CLOSURE_NATIVE,
        );

        let exception = badarity(
            &arc_process,
            function,
            arguments,
            Trace::capture(),
            Some(
                anyhow!(
                    "function arguments {} is {} term(s), but should be {}",
                    arguments,
                    arguments_len,
                    2,
                )
                .into(),
            ),
        );
        ErlangResult::error(arc_process.raise(exception))
    }
}
