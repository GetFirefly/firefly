//! ```elixir
//! case Lumen.Web.Element.set_attribute(element, "data-attribute", "data-value") do
//!   :ok -> ...
//!   {:error, {:name, name} -> ...
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use lumen_runtime::binary_to_string::binary_to_string;

use crate::element;

#[native_implemented_function(set_attribute/3)]
pub fn native(
    process: &Process,
    element_term: Term,
    name: Term,
    value: Term,
) -> exception::Result<Term> {
    let element = element::from_term(element_term)?;

    let name_string: String = binary_to_string(name)?;
    let value_string: String = binary_to_string(value)?;

    match element.set_attribute(&name_string, &value_string) {
        Ok(()) => Ok(atom!("ok")),
        // InvalidCharacterError JsValue
        Err(_) => {
            let name_tag = Atom::str_to_term("name");
            let reason = process.tuple_from_slice(&[name_tag, name])?;

            let error = atom!("error");

            process
                .tuple_from_slice(&[error, reason])
                .map_err(|error| error.into())
        }
    }
}
