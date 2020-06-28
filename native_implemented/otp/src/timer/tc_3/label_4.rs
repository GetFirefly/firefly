//! ```elixir
//! # label 4
//! # pushed to stack: (value)
//! # returned from call: duration
//! # full stack: (duration, value)
//! # returns: time
//! time = :erlang.convert_time_unit(duration, :native, :microsecond)
//! {time, value}
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::convert_time_unit_3;

use super::label_5;

// Private

#[native_implemented::label]
fn result(process: &Process, duration: Term, value: Term) -> Term {
    assert!(duration.is_integer());

    process.queue_frame_with_arguments(convert_time_unit_3::frame().with_arguments(
        false,
        &[
            duration,
            Atom::str_to_term("native"),
            Atom::str_to_term("microsecond"),
        ],
    ));
    process.queue_frame_with_arguments(label_5::frame().with_arguments(true, &[value]));

    Term::NONE
}
