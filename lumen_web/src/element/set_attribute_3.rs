use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::binary_to_string::binary_to_string;

use crate::{element, error, ok};

/// ```elixir
/// case Lumen.Web.Element.set_attribute(element, "data-attribute", "data-value") do
///   :ok -> ...
///   {:error, {:name, name} -> ...
/// end
/// ``
pub fn place_frame_with_arguments(
    process: &Process,
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

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let element = arc_process.stack_pop().unwrap();
    let attribute = arc_process.stack_pop().unwrap();
    let value = arc_process.stack_pop().unwrap();

    match native(arc_process, element, attribute, value) {
        Ok(ok_or_error) => {
            arc_process.return_from_call(ok_or_error)?;

            Process::call_code(arc_process)
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
        arity: 3,
    })
}

pub fn native(process: &Process, element_term: Term, name: Term, value: Term) -> exception::Result {
    let element = element::from_term(element_term)?;

    let name_string: String = binary_to_string(name)?;
    let value_string: String = binary_to_string(value)?;

    match element.set_attribute(&name_string, &value_string) {
        Ok(()) => Ok(ok()),
        // InvalidCharacterError JsValue
        Err(_) => {
            let name_tag = Atom::str_to_term("name");
            let reason = process.tuple_from_slice(&[name_tag, name])?;

            let error = error();

            process
                .tuple_from_slice(&[error, reason])
                .map_err(|error| error.into())
        }
    }
}
