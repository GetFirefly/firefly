use std::convert::TryInto;
use std::sync::Arc;

use web_sys::Element;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{resource, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::ok;

/// ```elixir
/// case Lumen.Web.Element.set_attribute(element, "data-attribute", "data-value") do
///   :ok -> ...
///   {:error, {:name, name} -> ...
/// end
/// ``
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    element: Term,
) -> Result<(), Alloc> {
    process.stack_push(element)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let element = arc_process.stack_pop().unwrap();

    match native(element) {
        Ok(ok) => {
            arc_process.return_from_call(ok)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("set_attribute").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(element: Term) -> exception::Result {
    let element_reference: resource::Reference = element.try_into()?;
    let element_element: &Element = element_reference.downcast_ref().ok_or_else(|| badarg!())?;

    element_element.remove();

    Ok(ok())
}
