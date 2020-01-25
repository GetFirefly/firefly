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
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::location::Location;

use locate_code::locate_code;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    let definition = Definition::Export {
        function: function(),
    };
    process.closure_with_env_from_slice(super::module(), definition, ARITY, Some(LOCATED_CODE), &[])
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
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let text = arc_process.stack_pop().unwrap();

    label_1::place_frame_with_arguments(arc_process, Placement::Replace, text).unwrap();
    lumen_web::window::window_0::place_frame_with_arguments(arc_process, Placement::Push).unwrap();

    Process::call_code(arc_process)
}

fn frame(location: Location, code: Code) -> Frame {
    Frame::new(super::module(), function(), ARITY, location, code)
}

fn function() -> Atom {
    Atom::try_from_str("dom_output").unwrap()
}
