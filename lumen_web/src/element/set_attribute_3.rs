use std::convert::TryInto;
use std::sync::Arc;

use web_sys::Element;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, resource, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::{error, ok};

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
    attribute: Term,
    value: Term,
) -> Result<(), Alloc> {
    process.stack_push(value)?;
    process.stack_push(attribute)?;
    process.stack_push(element)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let element = arc_process.stack_pop().unwrap();
    let attribute = arc_process.stack_pop().unwrap();
    let value = arc_process.stack_pop().unwrap();

    match native(arc_process, element, attribute, value) {
        Ok(ok_or_error) => {
            arc_process.return_from_call(ok_or_error)?;

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
        arity: 2,
    })
}

fn native(
    process: &ProcessControlBlock,
    element: Term,
    name: Term,
    value: Term,
) -> exception::Result {
    let element_reference: resource::Reference = element.try_into()?;
    let element_element: &Element = element_reference.downcast_ref().ok_or_else(|| badarg!())?;

    let name_string: String = name.try_into()?;
    let value_string: String = value.try_into()?;

    match element_element.set_attribute(&name_string, &value_string) {
        Ok(()) => Ok(ok()),
        // InvalidCharacterError JsValue
        Err(_) => {
            let name_tag = atom_unchecked("name");
            let reason = process.tuple_from_slice(&[name_tag, name])?;

            let error = error();

            process
                .tuple_from_slice(&[error, reason])
                .map_err(|error| error.into())
        }
    }
}
