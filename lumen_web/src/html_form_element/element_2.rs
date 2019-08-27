use std::convert::TryInto;
use std::sync::Arc;

use wasm_bindgen::JsCast;

use web_sys::HtmlInputElement;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::{error, html_form_element, ok};

/// ```elixir
/// case Lumen.Web.HTMLFormElement.element(html_form_element, "input-name") do
///   {:ok, html_input_element} -> ...
///   :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
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

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let html_form_element = arc_process.stack_pop().unwrap();
    let name = arc_process.stack_pop().unwrap();

    match native(arc_process, html_form_element, name) {
        Ok(ok_html_input_element_or_error) => {
            arc_process.return_from_call(ok_html_input_element_or_error)?;

            ProcessControlBlock::call_code(arc_process)
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

fn native(
    process: &ProcessControlBlock,
    html_form_element_term: Term,
    name: Term,
) -> exception::Result {
    lumen_runtime::system::io::puts(&format!(
        "[{}:{}] html_form_element_term = {:?}; name = {:?}",
        file!(),
        line!(),
        html_form_element_term,
        name
    ));
    let html_form_element_term = html_form_element::from_term(html_form_element_term)?;
    lumen_runtime::system::io::puts(&format!("[{}:{}]", file!(), line!(),));
    let name_string: String = name.try_into()?;
    lumen_runtime::system::io::puts(&format!("[{}:{}]", file!(), line!(),));

    let object = html_form_element_term.get_with_name(&name_string);
    let result_html_input_element: Result<HtmlInputElement, _> = object.dyn_into();

    match result_html_input_element {
        Ok(html_input_element) => {
            let html_input_element_resource_reference =
                process.resource(Box::new(html_input_element))?;

            process
                .tuple_from_slice(&[ok(), html_input_element_resource_reference])
                .map_err(|error| error.into())
        }
        Err(_) => Ok(error()),
    }
}
