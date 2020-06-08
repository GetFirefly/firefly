//! ```elixir
//! # label 2
//! # pushed to stack: (before)
//! # returned from call: value
//! # full stack: (value, before)
//! # returns: after
//! after = :erlang.monotonic_time()
//! duration = after - before
//! time = :erlang.convert_time_unit(duration, :native, :microsecond)
//! {time, value}
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::monotonic_time_0;

use super::label_3;

// Private

#[native_implemented::label]
fn result(process: &Process, value: Term, before: Term) -> Term {
    assert!(before.is_integer());

    process.queue_frame_with_arguments(monotonic_time_0::frame().with_arguments(false, &[]));
    process.queue_frame_with_arguments(label_3::frame().with_arguments(true, &[before, value]));

    Term::NONE
}
