//! Unlike [Node.appendChild](https://developer.mozilla.org/en-US/docs/Web/API/Node/appendChild),
//! this does not return the appended child as that is prone to errors with chaining.
//!
//! ```elixir
//! :ok = Lumen.Web.Node.append_child(parent, child)
//! ```
//!
//! It works with elements and can append elements or text nodes.
//!
//! ```elixir
//! div = Lumen.Web.Document.create_element(document, "div")
//! text = Lumen.Web.Document.create_text_node(document, "Text in the div")
//! Lumen.Web.Node.append_child(div, text)
//!
//! {:ok, window} = Lumen.Web.Window.window()
//! {:ok, document} = Lumen.Web.Window.document(document)
//! {:ok, element_with_id} = Lumen.Web.Document.get_element_by_id("element-id")
//! Lumen.Web.Node.append_child(element_with_id, div)
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{atom, badarg};

use lumen_runtime_macros::native_implemented_function;

use crate::node::node_from_term;

#[native_implemented_function(append_child/2)]
pub fn native(parent: Term, child: Term) -> exception::Result<Term> {
    let parent_node = node_from_term(parent)?;
    let child_node = node_from_term(child)?;

    // not sure how this could fail from `web-sys` or MDN docs.
    match parent_node.append_child(child_node) {
        Ok(_) => Ok(atom!("ok")),
        // JsValue(HierarchyRequestError: Failed to execute 'appendChild' on 'Node': The new child
        // element contains the parent.
        Err(_) => Err(badarg!().into()),
    }
}
