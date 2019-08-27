use std::convert::TryInto;
use std::sync::Arc;

use web_sys::Window;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, resource, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::otp::erlang;
use lumen_runtime::process::spawn::options::Options;

use crate::window::add_event_listener;

/// ```elixir
/// case Lumen.Web.Window.add_event_listener(window, :submit, MyModule, :my_function) do
///   :ok -> ...
///   :error -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    window: Term,
    event: Term,
    module: Term,
    function: Term,
) -> Result<(), Alloc> {
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.stack_push(event)?;
    process.stack_push(window)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let window = arc_process.stack_pop().unwrap();
    let event = arc_process.stack_pop().unwrap();
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();

    match native(window, event, module, function) {
        Ok(ok_or_error) => {
            arc_process.return_from_call(ok_or_error)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("add_event_listener").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 4,
    })
}

fn native(window: Term, event: Term, module: Term, function: Term) -> exception::Result {
    let window_reference: resource::Reference = window.try_into()?;
    let window_window: &Window = window_reference.downcast_ref().ok_or_else(|| badarg!())?;

    let event_atom: Atom = event.try_into()?;
    let _: Atom = module.try_into()?;
    let _: Atom = function.try_into()?;

    // TODO support passing in options to allow bigger heaps
    let options: Options = Default::default();

    add_event_listener(
        window_window,
        event_atom.name(),
        options,
        move |child_process, event_resource_reference| {
            let arguments = child_process.list_from_slice(&[event_resource_reference])?;

            erlang::apply_3::place_frame_with_arguments(
                child_process,
                Placement::Push,
                module,
                function,
                arguments,
            )
        },
    );

    Ok(atom_unchecked("ok"))
}
