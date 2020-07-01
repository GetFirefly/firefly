//! ```elixir
//! # label: 2
//! # pushed to stack: ()
//! # returned from call: {:ok, n_input}
//! # full stack: ({:ok, n_input})
//! # returns: value_string
//! value_string = Lumen.Web.HTMLInputElement.value(n_input)
//! n = :erlang.binary_to_integer(value_string)
//! dom(n)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_3;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_n_input: Term) -> Term {
    let ok_n_input_tuple: Boxed<Tuple> = ok_n_input
        .try_into()
        .unwrap_or_else(|_| panic!("ok_n_input ({:?}) is not a tuple", ok_n_input));
    assert_eq!(ok_n_input_tuple.len(), 2);
    assert_eq!(ok_n_input_tuple[0], Atom::str_to_term("ok"));
    let n_input = ok_n_input_tuple[1];
    assert!(n_input.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::html_input_element::value_1::frame().with_arguments(false, &[n_input]),
    );

    // ```elixir
    // # label: 3
    // # pushed to stack: ()
    // # returned from call: value_string
    // # full stack: (value_string)
    // # returns: n
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    process.queue_frame_with_arguments(label_3::frame().with_arguments(true, &[]));

    Term::NONE
}
