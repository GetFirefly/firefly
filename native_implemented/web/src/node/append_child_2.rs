//! Unlike [Node.appendChild](https://developer.mozilla.org/en-US/docs/Web/API/Node/appendChild),
//! this does not return the appended child as that is prone to errors with chaining.
//!
//! ```elixir
//! case Lumen.Web.Node.append_child(parent, child) do
//!   :ok -> ...
//!   {:error, reason} -> ...
//! end
//! ```
//!
//! It works with elements and can append elements or text nodes.
//!
//! ```elixir
//! div = Lumen.Web.Document.create_element(document, "div")
//! text = Lumen.Web.Document.create_text_node(document, "Text in the div")
//! :ok = Lumen.Web.Node.append_child(div, text)
//!
//! {:ok, window} = Lumen.Web.Window.window()
//! {:ok, document} = Lumen.Web.Window.document(document)
//! {:ok, element_with_id} = Lumen.Web.Document.get_element_by_id("element-id")
//! Lumen.Web.Node.append_child(element_with_id, div)
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::{error_tuple, node};

#[native_implemented::function(Elixir.Lumen.Web.Node:append_child/2)]
pub fn result(process: &Process, parent: Term, child: Term) -> exception::Result<Term> {
    let parent_node = node::from_term(parent)?;
    let child_node = node::from_term(child)?;

    let final_term = match parent_node.append_child(child_node) {
        Ok(_) => atom!("ok"),
        // JsValue(HierarchyRequestError: Failed to execute 'appendChild' on 'Node': The new child
        // element contains the parent.
        Err(js_value) => error_tuple(process, js_value),
    };

    Ok(final_term)
}
