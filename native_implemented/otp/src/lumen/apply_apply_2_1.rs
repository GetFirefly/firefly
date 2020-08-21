//! This is used as the `init_fn` for `Scheduler::spawn_closure`, as the spawning code can only
//! pass at most 1 argument, the closure itself to the `init_fn`'s native code and `erlang:apply/2`
//! takes two arguments

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badarity};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;

#[native_implemented::function(lumen:apply_apply_2/1)]
fn result(process: &Process, arguments: Term) -> exception::Result<Term> {
    let environment_boxed_cons: Boxed<Cons> = term_try_into_non_empty_list!(arguments)?;

    let mut environment_vec: Vec<Term> = Vec::new();

    for result in environment_boxed_cons.into_iter() {
        match result {
            Ok(element) => environment_vec.push(element),
            Err(_) => {
                return Err(anyhow!(ImproperListError))
                    .context(format!(
                        "arguments ({}) is not a proper list",
                        arguments
                    ))
                    .map_err(From::from)
            }
        }
    }

    let environment_len = environment_vec.len();

    if environment_len == (erlang::apply_2::ARITY as usize) {
        let function = environment_vec[0];
        let arguments = environment_vec[1];

        // want to call into `erlang:apply/2` `native` and not `result` so that the stack trace
        // shows `erlang:apply/2`.
        Ok(erlang::apply_2::native(function, arguments))
    } else {
        let function = process.anonymous_closure_with_env_from_slice(
            super::module(),
            Default::default(),
            Default::default(),
            Default::default(),
            0,
            CLOSURE_NATIVE,
            process.pid().into(),
            &environment_vec,
        );

        Err(badarity(
            process,
            function,
            Term::NIL,
            anyhow!(
                "function arguments {} is {} term(s), but should be {}",
                arguments,
                environment_len,
                erlang::apply_2::ARITY
            )
            .into(),
        ))
    }
}
