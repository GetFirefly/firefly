//! ```elixir
//! class_name = Lumen.Web.Element.class_name(element)
//! ``

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::element;

#[native_implemented_function(class_name/1)]
fn native(process: &Process, element_term: Term) -> exception::Result<Term> {
    let element = element::from_term(process, element_term)?;
    let class_name_binary = process.binary_from_str(&element.class_name())?;

    Ok(class_name_binary)
}
