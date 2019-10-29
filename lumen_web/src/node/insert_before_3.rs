use std::sync::Arc;

use wasm_bindgen::JsCast;

use web_sys::DomException;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::node::node_from_term;
use crate::ok_tuple;

/// Unlike [Node.appendChild](https://developer.mozilla.org/en-US/docs/Web/API/Node/appendChild),
/// this does not return the appended child as that is prone to errors with chaining.
///
/// ```elixir
/// case Lumen.Web.Node.insert_before(parent, new_child, reference_child) do
///   {:ok, inserted_child} -> ...
///   {:error, :hierarchy_request} -> ...
///   {:error, :wrong_document} -> ...
///   {:error, :no_modification_allowed} -> ...
///   {:error, :not_found} -> ...
///   {:error, :not_supported} -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    parent: Term,
    new_child: Term,
    reference_child: Term,
) -> Result<(), Alloc> {
    process.stack_push(reference_child)?;
    process.stack_push(new_child)?;
    process.stack_push(parent)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

pub fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let parent = arc_process.stack_pop().unwrap();
    let new_child = arc_process.stack_pop().unwrap();
    let reference_child = arc_process.stack_pop().unwrap();

    match native(arc_process, parent, new_child, reference_child) {
        Ok(ok_or_error_tuple) => {
            arc_process.return_from_call(ok_or_error_tuple)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("insert_before").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}

fn native(
    process: &Process,
    parent: Term,
    new_child: Term,
    reference_child: Term,
) -> exception::Result<Term> {
    let parent_node = node_from_term(parent)?;
    let new_child_node = node_from_term(new_child)?;

    let option_reference_child_node = if reference_child == Atom::str_to_term("nil") {
        None
    } else {
        Some(node_from_term(reference_child)?)
    };

    match parent_node.insert_before(new_child_node, option_reference_child_node) {
        Ok(inserted_child_node) => {
            ok_tuple(process, Box::new(inserted_child_node)).map_err(|error| error.into())
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
            let reason = Atom::str_to_term(reason_name);
            let error_tuple = process.tuple_from_slice(&[atom!("error"), reason])?;

            Ok(error_tuple)
        }
    }
}
