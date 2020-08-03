//! ```elixir
//! # pushed to stack: (first, last, acc, reducer)
//! # returned from call: new_first
//! # full stack: (new_first, first, last, acc, reducer)
//! # returns: new_acc
//! new_acc = reducer.(first, acc)
//! reduce_range_inc(new_first, last, new_acc, reducer)
//! ```

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::*;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

#[native_implemented::label]
fn result(
    process: &Process,
    new_first: Term,
    first: Term,
    last: Term,
    acc: Term,
    reducer: Term,
) -> exception::Result<Term> {
    assert!(new_first.is_integer());
    assert!(first.is_integer());
    assert!(last.is_integer());

    match reducer.decode().unwrap() {
        TypedTerm::Closure(closure) => {
            if closure.arity() == 2 {
                process.queue_frame_with_arguments(
                    closure.frame_with_arguments(false, vec![first, acc]),
                );

                process.queue_frame_with_arguments(
                    label_2::frame().with_arguments(true, &[new_first, last, reducer]),
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
