use std::sync::Arc;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::node::node_from_term;
use crate::ok;

/// Unlike [Node.appendChild](https://developer.mozilla.org/en-US/docs/Web/API/Node/appendChild),
/// this does not return the appended child as that is prone to errors with chaining.
///
/// ```elixir
/// :ok = Lumen.Web.Node.append_child(parent, child)
/// ```
///
/// It works with elements and can append elements or text nodes.
///
/// ```elixir
/// div = Lumen.Web.Document.create_element(document, "div")
/// text = Lumen.Web.Document.create_text_node(document, "Text in the div")
/// Lumen.Web.Node.append_child(div, text)
///
/// {:ok, window} = Lumen.Web.Window.window()
/// {:ok, document} = Lumen.Web.Window.document(document)
/// {:ok, element_with_id} = Lumen.Web.Document.get_element_by_id("element-id")
/// Lumen.Web.Node.append_child(element_with_id, div)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    parent: Term,
    child: Term,
) -> Result<(), Alloc> {
    process.stack_push(child)?;
    process.stack_push(parent)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

pub fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let parent = arc_process.stack_pop().unwrap();
    let child = arc_process.stack_pop().unwrap();

    match native(parent, child) {
        Ok(ok) => {
            arc_process.return_from_call(ok)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("append_child").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub fn native(parent: Term, child: Term) -> exception::Result {
    let parent_node = node_from_term(parent)?;
    let child_node = node_from_term(child)?;

    // not sure how this could fail from `web-sys` or MDN docs.
    match parent_node.append_child(child_node) {
        Ok(_) => Ok(ok()),
        // JsValue(HierarchyRequestError: Failed to execute 'appendChild' on 'Node': The new child
        // element contains the parent.
        Err(_) => Err(badarg!().into()),
    }
}
