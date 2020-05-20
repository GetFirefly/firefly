//! ```elixir
//! case Lumen.Web.Element.set_attribute(element, "data-attribute", "data-value") do
//!   :ok -> ...
//!   {:error, {:name, name} -> ...
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::element;

#[native_implemented_function(remove/1)]
fn result(element_term: Term) -> exception::Result<Term> {
    let element = element::from_term(element_term)?;

    element.remove();

    Ok(atom!("ok"))
}
