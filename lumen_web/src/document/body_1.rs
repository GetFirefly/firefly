use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::document::document_from_term;
use crate::option_to_ok_tuple_or_error;

/// ```elixir
/// case Lumen.Web.Document.body(document) do
///   {:ok, body} -> ...
///   :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
) -> Result<(), Alloc> {
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let document = arc_process.stack_pop().unwrap();

    match native(arc_process, document) {
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
    Atom::try_from_str("body").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

pub fn native(process: &Process, document: Term) -> exception::Result<Term> {
    let document_document = document_from_term(document)?;

    option_to_ok_tuple_or_error(process, document_document.body()).map_err(|error| error.into())
}
