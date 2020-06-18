//! ```elixir
//! value_string = Lumen.Web.HTMLInputElement.value(html_input_element)
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::html_input_element;

#[native_implemented::function(Elixir.Lumen.Web.HTMLInputElement:value/1)]
fn result(process: &Process, html_input_element_term: Term) -> exception::Result<Term> {
    let html_input_element = html_input_element::from_term(html_input_element_term)?;
    let value_string = html_input_element.value();

    process
        .binary_from_str(&value_string)
        .map_err(|error| error.into())
}
