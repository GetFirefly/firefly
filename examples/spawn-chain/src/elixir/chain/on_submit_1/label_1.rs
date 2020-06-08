//! ```elixir
//! # label: 1
//! # pushed to stack: ()
//! # returned from call: {:ok, event_target}
//! # full stack: ({:ok, event_target})
//! # returns: {:ok, n_input}
//! {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
//! value_string = Lumen.Web.HTMLInputElement.value(n_input)
//! n = :erlang.binary_to_integer(value_string)
//! dom(n)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

// Private

#[native_implemented::label]
fn result(process: &Process, ok_event_target: Term) -> exception::Result<Term> {
    assert!(
        ok_event_target.is_boxed_tuple(),
        "ok_event_target ({:?}) is not a tuple",
        ok_event_target
    );
    let ok_event_target_tuple: Boxed<Tuple> = ok_event_target.try_into().unwrap();
    assert_eq!(ok_event_target_tuple.len(), 2);
    assert_eq!(ok_event_target_tuple[0], Atom::str_to_term("ok"));
    let event_target = ok_event_target_tuple[1];
    assert!(event_target.is_boxed_resource_reference());

    let name = process.binary_from_str("n")?;
    process.queue_frame_with_arguments(
        liblumen_web::html_form_element::element_2::frame()
            .with_arguments(false, &[event_target, name]),
    );

    // ```elixir
    // # label: 2
    // # pushed to stack: ()
    // # returned from call: {:ok, n_input}
    // # full stack: ({:ok, n_input})
    // # returns: value_string
    // value_string = Lumen.Web.HTMLInputElement.value(n_input)
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    process.queue_frame_with_arguments(label_2::frame().with_arguments(true, &[]));

    Ok(Term::NONE)
}
