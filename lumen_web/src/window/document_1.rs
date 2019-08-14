use std::convert::TryInto;
use std::sync::Arc;

use web_sys::Window;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{resource, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::option_to_ok_tuple_or_error;

/// ```elixir
/// case Lumen.Web.Document.document(window) do
///    {:ok, document} -> ...
///    :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    window: Term,
) -> Result<(), Alloc> {
    process.stack_push(window)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let window = arc_process.stack_pop().unwrap();

    match native(arc_process, window) {
        Ok(ok_tuple_or_error) => {
            arc_process.return_from_call(ok_tuple_or_error)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("document").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &ProcessControlBlock, window: Term) -> exception::Result {
    let window_reference: resource::Reference = window.try_into()?;
    let window_window: &Window = window_reference.downcast_ref().ok_or_else(|| badarg!())?;
    let option_document = window_window.document();

    option_to_ok_tuple_or_error(process, option_document).map_err(|error| error.into())
}
