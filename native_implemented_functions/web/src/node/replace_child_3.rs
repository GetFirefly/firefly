//! Unlike [Node.appendChild](https://developer.mozilla.org/en-US/docs/Web/API/Node/appendChild),
//! this does not return the appended child as that is prone to errors with chaining.
//!
//! ```elixir
//! case Lumen.Web.Node.replace_child(parent, new_child, old_child) do
//!   {:ok, replaced_child} -> ...
//!   {:error, :hierarchy_request} -> ...
//!   {:error, :wrong_document} -> ...
//!   {:error, :no_modification_allowed} -> ...
//!   {:error, :not_found} -> ...
//!   {:error, :not_supported} -> ...
//! end
//! ```

use wasm_bindgen::JsCast;

use web_sys::DomException;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::node::node_from_term;
use crate::ok_tuple;

#[native_implemented_function(replace_child/3)]
fn result(
    process: &Process,
    parent: Term,
    old_child: Term,
    new_child: Term,
) -> exception::Result<Term> {
    let parent_node = node_from_term(parent)?;
    let old_child_node = node_from_term(old_child)?;
    let new_child_node = node_from_term(new_child)?;

    match parent_node.replace_child(old_child_node, new_child_node) {
        Ok(replaced_child_node) => {
            ok_tuple(process, Box::new(replaced_child_node)).map_err(|error| error.into())
        }
        Err(js_value) => {
            let dom_exception = js_value.dyn_into::<DomException>().unwrap();

            let reason_name = match dom_exception.name().as_str() {
                "HierarchyRequestError" => "hierarchy_request",
                "NoModificationAllowedError" => "no_modification_allowed",
                "NotFoundError" => "not_found",
                "NotSupportedError" => "not_supported",
                "WrongDocumentError" => "wrong_document",
                name => unimplemented!(
                    "Convert DOMException with name ({:?}) to error reason",
                    name
                ),
            };
            let reason = atom!(reason_name);
            let error_tuple = process.tuple_from_slice(&[atom!("error"), reason])?;

            Ok(error_tuple)
        }
    }
}
