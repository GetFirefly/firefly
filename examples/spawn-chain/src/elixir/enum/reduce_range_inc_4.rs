//! ```elixir
//! defp reduce_range_inc(first, first, acc, fun) do
//!   fun.(first, acc)
//! end
//!
//! defp reduce_range_inc(first, last, acc, fun) do
//!   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
//! end
//! ```

mod label_1;
mod label_2;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::*;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang::add_2;

#[native_implemented::function(reduce_range_inc/4)]
fn result(
    process: &Process,
    first: Term,
    last: Term,
    acc: Term,
    reducer: Term,
) -> exception::Result<Term> {
    // defp reduce_range_inc(first, first, acc, fun) do
    //   fun.(first, acc)
    // end
    if first == last {
        match reducer.decode().unwrap() {
            TypedTerm::Closure(closure) => {
                if closure.arity() == 2 {
                    process.queue_frame_with_arguments(
                        closure.frame_with_arguments(false, vec![first, acc]),
                    );

                    Ok(Term::NONE)
                } else {
                    let argument_list = process.list_from_slice(&[first, acc])?;

                    Err(badarity(
                        process,
                        reducer,
                        argument_list,
                        anyhow!("reducer").into(),
                    ))
                }
            }
            _ => Err(badfun(process, reducer, anyhow!("reducer").into())),
        }
    }
    // defp reduce_range_inc(first, last, acc, fun) do
    //   reduce_range_inc(first + 1, last, fun.(first, acc), fun)
    // end
    else {
        // ```elixir
        // # pushed to stack: (first, inc)
        // # returned from call: N/A
        // # full stack: (first, inc)
        // # returns: new_first
        // first + 1
        // ```
        let inc = process.integer(1)?;
        process.queue_frame_with_arguments(add_2::frame().with_arguments(false, &[first, inc]));

        // ```elixir
        // # pushed to stack: (first, last, acc, reducer)
        // # returned from call: new_first
        // # full stack: (new_first, first, last, acc, reducer)
        // # returns: new_acc
        // new_acc = reducer.(first, acc)
        // reduce_range_inc(new_first, last, new_acc, reducer)
        // ```
        process.queue_frame_with_arguments(
            label_1::frame().with_arguments(true, &[first, last, acc, reducer]),
        );

        Ok(Term::NONE)
    }
}
