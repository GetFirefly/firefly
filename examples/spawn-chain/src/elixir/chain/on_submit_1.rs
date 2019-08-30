mod label_1;
mod label_2;
mod label_3;
mod label_4;

use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

/// ```elixir
/// # pushed to stack: (event)
/// # returned from call: N/A
/// # full stack: (event)
/// # returns: {:ok, event_target}
/// def on_submit(event) do
///   {:ok, event_target} = Lumen.Web.Event.target(event)
///   {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
///   value_string = Lumen.Web.HTMLInputElement.value(n_input)
///   n = :erlang.binary_to_integer(value_string)
///   dom(n)
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    event: Term,
) -> Result<(), Alloc> {
    assert!(event.is_resource_reference());
    process.stack_push(event)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let event = arc_process.stack_pop().unwrap();
    assert!(event.is_resource_reference());

    // ```elixir
    // # label: 1
    // # pushed to stack: ()
    // # returned from call: {:ok, event_target}
    // # full stack: ({:ok, event_target})
    // # returns: {:ok, n_input}
    // {:ok, n_input} = Lumen.Web.HTMLFormElement.element(event_target, "n")
    // value_string = Lumen.Web.HTMLInputElement.value(n_input)
    // n = :erlang.binary_to_integer(value_string)
    // dom(n)
    // ```
    label_1::place_frame(arc_process, Placement::Replace);

    lumen_web::event::target_1::place_frame_with_arguments(arc_process, Placement::Push, event)?;

    Process::call_code(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("on_submit").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}
