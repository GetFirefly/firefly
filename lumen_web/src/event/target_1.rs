use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::{error, event, ok};

/// ```elixir
/// case Lumen.Web.Event.target(event) do
///
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    event: Term,
) -> Result<(), Alloc> {
    process.stack_push(event)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let event = arc_process.stack_pop().unwrap();

    match native(arc_process, event) {
        Ok(ok_event_target_or_error) => {
            arc_process.return_from_call(ok_event_target_or_error)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("target").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

fn native(process: &ProcessControlBlock, event_term: Term) -> exception::Result {
    let event = event::from_term(event_term)?;

    match event.target() {
        Some(event_target) => {
            let event_target_resource_reference = process.resource(Box::new(event_target))?;

            process
                .tuple_from_slice(&[ok(), event_target_resource_reference])
                .map_err(|error| error.into())
        }
        None => Ok(error()),
    }
}
