//! ```elixir
//! # label 3
//! # pushed to stack: (before, value)
//! # returned from call: after
//! # full stack: (after, before, value)
//! # returns: duration
//! duration = after - before
//! time = :erlang.convert_time_unit(duration, :native, :microsecond)
//! {time, value}
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::subtract_2;

use super::label_4;

// Private

#[native_implemented::label]
fn result(process: &Process, after: Term, before: Term, value: Term) -> Term {
    assert!(after.is_integer());
    assert!(before.is_integer());

    process.queue_frame_with_arguments(subtract_2::frame().with_arguments(false, &[after, before]));
    process.queue_frame_with_arguments(label_4::frame().with_arguments(true, &[value]));

    Term::NONE
}
