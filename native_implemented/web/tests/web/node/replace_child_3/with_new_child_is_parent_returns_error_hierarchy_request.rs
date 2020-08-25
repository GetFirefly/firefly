//! ```elixir
//! {:ok, document} = Lumen.Web.Document.new()
//! {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
//! {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
//! :ok = Lumen.Web.Node.append_child(parent, old_child)
//! {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
//! ```

#[path = "with_new_child_is_parent_returns_error_hierarchy_request/label_1.rs"]
pub mod label_1;
#[path = "with_new_child_is_parent_returns_error_hierarchy_request/label_2.rs"]
pub mod label_2;
#[path = "with_new_child_is_parent_returns_error_hierarchy_request/label_3.rs"]
pub mod label_3;
#[path = "with_new_child_is_parent_returns_error_hierarchy_request/label_4.rs"]
pub mod label_4;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::document;

#[native_implemented::function(Elixir.Lumen.Web.Node.ReplaceChild3:with_new_child_is_parent_returns_error_hierarchy_request/0)]
fn result(process: &Process) -> Term {
    // ```elixir
    // # pushed to stack: ()
    // # returned from call: N/A
    // # full stack: ()
    // # returns: {:ok, document}
    // ```
    process.queue_frame_with_arguments(document::new_0::frame().with_arguments(false, &[]));
    // ```elixir
    // # label 1
    // # pushed to stack: ()
    // # returned form call: {:ok, document}
    // # full stack: ({:ok, document})
    // # returns: {:ok parent}
    // {:ok, old_child} = Lumen.Web.Document.create_element(document, "table")
    // {:ok, parent} = Lumen.Web.Document.create_element(document, "div")
    // :ok = Lumen.Web.Node.append_child(parent, old_child)
    // {:error, :hierarchy_request} = Lumen.Web.replace_child(parent, old_child, parent)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}
