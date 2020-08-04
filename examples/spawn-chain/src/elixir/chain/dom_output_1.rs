//! ```elixir
//! # pushed to stack: (text)
//! # returned from call: N/A
//! # full stack: (text)
//! defp dom_output(text) do
//!   {:ok, window} = Lumen::Web::Window.window()
//!   {:ok, document} = Lumen::Web::Window.document(window)
//!   {:ok, tr} = Lumen::Web::Document.create_element(document, "tr")
//!
//!   {:ok, pid_text} = Lumen::Web::Document.create_text_node(document, to_string(self()))
//!   {:ok, pid_td} = Lumen::Web::Document.create_element(document, "td")
//!   Lumen::Web::Node.append_child(pid_td, pid_text);
//!   Lumen::Web::Node.append_child(tr, pid_td)
//!
//!   {:ok, text_text} = Lumen::Web::Document.create_text_node(document, to_string(text()))
//!   {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
//!   Lumen::Web::Node.append_child(text_td, text_text);
//!   Lumen::Web::Node.append_child(tr, text_td)
//!
//!   {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
//!   Lumen::Web::Node.append_child(tbody, tr)
//! end
//! ```

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

use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn closure(process: &Process) -> Result<Term, Alloc> {
    process.export_closure(super::module(), function(), ARITY, CLOSURE_NATIVE)
}

// Private

#[native_implemented::function(Elixir.Chain:dom_output/1)]
fn result(process: &Process, text: Term) -> Term {
    process.queue_frame_with_arguments(
        liblumen_web::window::window_0::frame().with_arguments(false, &[]),
    );
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[text]));

    Term::NONE
}
