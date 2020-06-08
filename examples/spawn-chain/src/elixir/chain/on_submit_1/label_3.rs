//! ```elixir
//! # label: 3
//! # pushed to stack: ()
//! # returned from call: value_string
//! # full stack: (value_string)
//! # returns: n
//! n = :erlang.binary_to_integer(value_string)
//! dom(n)
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

use super::label_4;

// Private

#[native_implemented::label]
fn result(process: &Process, value_string: Term) -> Term {
    assert!(value_string.is_binary());

    process.queue_frame_with_arguments(
        erlang::binary_to_integer_1::frame().with_arguments(false, &[value_string]),
    );

    // ```elixir
    // # label: 4
    // # pushed to stack: ()
    // # returned from call: n
    // # full stack: (n)
    // # returns: {time, value}
    // dom(n)
    // ```
    process.queue_frame_with_arguments(label_4::frame().with_arguments(true, &[]));

    Term::NONE
}
