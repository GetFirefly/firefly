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

use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::Placement;
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Arity;

pub fn export() {
    lumen_rt_full::code::export::insert(super::module(), function(), ARITY, code);
}

// Private

const ARITY: Arity = 1;

fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let event = arc_process.stack_pop().unwrap();
    assert!(event.is_boxed_resource_reference());

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
    label_1::place_frame(arc_process, Placement::Replace);

    liblumen_web::event::target_1::place_frame_with_arguments(arc_process, Placement::Push, event)
        .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("on_submit").unwrap()
}
