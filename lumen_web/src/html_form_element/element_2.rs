//! ```elixir
//! case Lumen.Web.HTMLFormElement.element(html_form_element, "input-name") do
//!   {:ok, html_input_element} -> ...
//!   :error -> ...
//! end
//! ```

use wasm_bindgen::JsCast;

use web_sys::HtmlInputElement;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use lumen_runtime::binary_to_string::binary_to_string;

use crate::{error, html_form_element, ok};

#[native_implemented_function(element/2)]
fn native(process: &Process, html_form_element_term: Term, name: Term) -> exception::Result {
    let html_form_element_term = html_form_element::from_term(html_form_element_term)?;
    let name_string: String = binary_to_string(name)?;

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
