mod label_1;
mod label_10;
mod label_11;
mod label_12;
mod label_2;
mod label_3;
mod label_4;
mod label_5;
mod label_6;
mod label_7;
mod label_8;
mod label_9;

use std::sync::Arc;

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::frames::stack::frame::Placement;
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(super::module(), function(), ARITY, Some(code))
}

// Private

const ARITY: u8 = 1;

/// ```elixir
/// # pushed to stack: (text)
/// # returned from call: N/A
/// # full stack: (text)
/// defp dom_output(text) do
///   {:ok, window} = Lumen::Web::Window.window()
///   {:ok, document} = Lumen::Web::Window.document(window)
///   {:ok, tr} = Lumen::Web::Document.create_element(document, "tr")
///
///   {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
///   {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
///   Lumen::Web::Node.append_child(pid_td, pid_text);
///   Lumen::Web::Node.append_child(tr, pid_td)
///
///   {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
///   {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
///   Lumen::Web::Node.append_child(text_td, text_text);
///   Lumen::Web::Node.append_child(tr, text_td)
///
///   {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
///   Lumen::Web::Node.append_child(tbody, tr)
/// end
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let text = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, text).unwrap();
    liblumen_web::window::window_0::place_frame_with_arguments(arc_process, Placement::Push)
        .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("dom_output").unwrap()
}
