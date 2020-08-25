//! ```elixir
//! {:ok, document} = Lumen.Web.Document.new()
//! {:ok, reference_child} = Lumen.Web.Document.create_element(document, "table")
//! {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
//! :ok = Lumen.Web.Node.append_child(document, parent)
//! :ok = Lumen.Web.Node.append_child(parent, reference_child)
//! {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
//! {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
//! ```

#[path = "with_reference_child_inserts_before_reference_child/label_1.rs"]
pub mod label_1;
#[path = "with_reference_child_inserts_before_reference_child/label_2.rs"]
pub mod label_2;
#[path = "with_reference_child_inserts_before_reference_child/label_3.rs"]
pub mod label_3;
#[path = "with_reference_child_inserts_before_reference_child/label_4.rs"]
pub mod label_4;
#[path = "with_reference_child_inserts_before_reference_child/label_5.rs"]
pub mod label_5;
#[path = "with_reference_child_inserts_before_reference_child/label_6.rs"]
pub mod label_6;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::document;

#[native_implemented::function(Elixir.Lumen.Web.Node.InsertBefore3:with_reference_child_inserts_before_reference_child/0)]
fn result(process: &Process) -> Term {
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
    // # returns: {:ok, old_child}
    // {:ok, reference_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(parent_document, "div")
    // :ok = Lumen.Web.Node.append_child(document, parent)
    // :ok = Lumen.Web.Node.append_child(parent, reference_child)
    // {:ok, new_child} = Lumen.Web.Document.create_element(document, "ul");
    // {:ok, inserted_child} = Lumen.Web.insert_before(parent, new_child, reference_child)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}
