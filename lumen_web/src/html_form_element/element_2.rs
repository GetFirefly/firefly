use std::sync::Arc;

use wasm_bindgen::JsCast;

use web_sys::HtmlInputElement;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::binary_to_string::binary_to_string;

use crate::html_form_element;

/// ```elixir
/// case Lumen.Web.HTMLFormElement.element(html_form_element, "input-name") do
///   {:ok, html_input_element} -> ...
///   :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    html_form_element: Term,
    name: Term,
) -> Result<(), Alloc> {
    process.stack_push(name)?;
    process.stack_push(html_form_element)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let html_form_element = arc_process.stack_pop().unwrap();
    let name = arc_process.stack_pop().unwrap();

    match native(arc_process, html_form_element, name) {
        Ok(ok_html_input_element_or_error) => {
            arc_process.return_from_call(ok_html_input_element_or_error)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("element").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

fn native(process: &Process, html_form_element_term: Term, name: Term) -> exception::Result<Term> {
    let html_form_element_term = html_form_element::from_term(html_form_element_term)?;
    let name_string: String = binary_to_string(name)?;

    let object = html_form_element_term.get_with_name(&name_string);
    let result_html_input_element: Result<HtmlInputElement, _> = object.dyn_into();

    match result_html_input_element {
        Ok(html_input_element) => {
            let html_input_element_resource_reference =
                process.resource(Box::new(html_input_element))?;

            process
                .tuple_from_slice(&[atom!("ok"), html_input_element_resource_reference])
                .map_err(|error| error.into())
        }
        Err(_) => Ok(atom!("error")),
    }
}
