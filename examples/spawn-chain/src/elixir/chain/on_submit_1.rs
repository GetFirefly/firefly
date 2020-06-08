//! ```elixir
//! # pushed to stack: (event)
//! # returned from call: N/A
//! # full stack: (event)
//! # returns: {:ok, event_target}
//! def on_submit(event) do
//!   {:ok, event_target} = Lumen.Web.Event.target(event)
//!   {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
//!   value_string = Lumen.Web.HTMLInputElement.value(n_input)
//!   n = :erlang.binary_to_integer(value_string)
//!   dom(n)
//! end
//! ```

mod label_1;
mod label_2;
mod label_3;
mod label_4;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

// Private

#[native_implemented::function(on_submit/1)]
fn result(process: &Process, event: Term) -> Term {
    assert!(event.is_boxed_resource_reference());

    process.queue_frame_with_arguments(
        liblumen_web::event::target_1::frame().with_arguments(false, &[event]),
    );

    // ```elixir
    // # label: 1
    // # pushed to stack: ()
    // # returned from call: {:ok, event_target}
    // # full stack: ({:ok, event_target})
    // # returns: {:ok, n_input}
    // {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
    // value_string = Lumen.Web.HTMLInputElement.value(n_input)
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}
