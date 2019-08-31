use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::document::document_from_term;
use crate::option_to_ok_tuple_or_error;

/// ```elixir
/// case Lumen.Web.Document.get_element_by_id(document, "element-id") do
///   {:ok, element} -> ...
///   :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    id: Term,
) -> Result<(), Alloc> {
    process.stack_push(id)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let document = arc_process.stack_pop().unwrap();
    let id = arc_process.stack_pop().unwrap();

    match native(arc_process, document, id) {
        Ok(ok_tuple_or_error) => {
            arc_process.return_from_call(ok_tuple_or_error)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("get_element_by_id").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

fn native(process: &Process, document: Term, id: Term) -> exception::Result {
    let document_document = document_from_term(document)?;
    let id_string: String = id.try_into()?;

    option_to_ok_tuple_or_error(process, document_document.get_element_by_id(&id_string))
        .map_err(|error| error.into())
}
