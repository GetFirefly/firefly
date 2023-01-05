//! ```elixir
//! {:ok, document} = Lumen.Web.Document.new()
//! {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
//! {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
//! ```

#[path = "with_new_child_returns_ok_replaced_child/label_1.rs"]
pub mod label_1;
#[path = "with_new_child_returns_ok_replaced_child/label_2.rs"]
pub mod label_2;
#[path = "with_new_child_returns_ok_replaced_child/label_3.rs"]
pub mod label_3;
#[path = "with_new_child_returns_ok_replaced_child/label_4.rs"]
pub mod label_4;
#[path = "with_new_child_returns_ok_replaced_child/label_5.rs"]
pub mod label_5;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::document;

#[native_implemented::function(Elixir.Lumen.Web.Node.ReplaceChild3:with_new_child_returns_ok_replaced_child/0)]
pub fn result(process: &Process) -> Term {
    // ```elixir
    // # pushed to stack: ()
    // # returned from call: N/A
    // # full stack: ()
    // # returns: {:ok, parent_document}
    // ```
    process.queue_frame_with_arguments(document::new_0::frame().with_arguments(false, &[]));
    // ```elixir
    // # label 1
    // # pushed to stack: ()
    // # returned form call: {:ok, document}
    // # full stack: ({:ok, document})
    // # returns: {:ok parent}
    // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
    // :ok = Lumen.Web.Node.append_child(parent, old_child)
    // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
    // {:ok, replaced_child} = Lumen.Web.replace_child(parent, new_child, old_child)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}
