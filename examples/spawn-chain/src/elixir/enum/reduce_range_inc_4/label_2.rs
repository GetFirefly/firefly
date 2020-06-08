//! ```elixir
//! # pushed to stack: (new_first, last, reducer)
//! # returned from call: new_acc
//! # full stack: (new_acc, last, reducer)
//! # returns: final
//! reduce_range_inc(new_first, last, new_acc, reducer)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::elixir::r#enum::reduce_range_inc_4;

#[native_implemented::label]
fn result(process: &Process, new_acc: Term, new_first: Term, last: Term, reducer: Term) -> Term {
    // new_acc is on top of stack because it is the return from `reducer` call

    process.queue_frame_with_arguments(
        reduce_range_inc_4::frame().with_arguments(false, &[new_first, last, new_acc, reducer]),
    );

    Term::NONE
}
