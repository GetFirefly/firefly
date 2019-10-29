use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::html_input_element;

/// ```elixir
/// value_string = Lumen.Web.HTMLInputElement.value(html_input_element)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    html_input_element: Term,
) -> Result<(), Alloc> {
    process.stack_push(html_input_element)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let html_input_element = arc_process.stack_pop().unwrap();

    match native(arc_process, html_input_element) {
        Ok(body) => {
            arc_process.return_from_call(body)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("value").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &Process, html_input_element_term: Term) -> exception::Result<Term> {
    let html_input_element = html_input_element::from_term(html_input_element_term)?;
    let value_string = html_input_element.value();

    process
        .binary_from_str(&value_string)
        .map_err(|error| error.into())
}
