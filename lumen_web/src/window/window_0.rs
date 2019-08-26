use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Atom;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::option_to_ok_tuple_or_error;

/// ```elixir
/// case Lumen.Web.Window.window() do
///    {:ok, window} -> ...
///    :error -> ...
/// end
/// ```
pub fn place_frame(process: &ProcessControlBlock, placement: Placement) {
    process.place_frame(frame(), placement);
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    match native(arc_process) {
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
    Atom::try_from_str("window").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

pub fn native(process: &ProcessControlBlock) -> exception::Result {
    let option_window = web_sys::window();

    option_to_ok_tuple_or_error(process, option_window).map_err(|error| error.into())
}
