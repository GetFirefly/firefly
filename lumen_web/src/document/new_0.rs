use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Atom;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::{error, ok_tuple};

/// Returns a new document whose `origin` is the `origin` of the current global object's associated
/// `Document`.
///
/// ```elixir
/// case Lumen.Web.Document.new() do
///    {:ok, document} -> ...
///    :error -> ...
/// end
/// ```
pub fn place_frame(process: &Process, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    match native(arc_process) {
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
    Atom::try_from_str("new").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 0,
    })
}

fn native(process: &Process) -> exception::Result {
    match web_sys::Document::new() {
        Ok(document) => ok_tuple(process, Box::new(document)).map_err(|error| error.into()),
        // Not sure how this can happen
        Err(_) => Ok(error()),
    }
}
