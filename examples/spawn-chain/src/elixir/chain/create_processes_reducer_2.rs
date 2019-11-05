use std::sync::Arc;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use lumen_runtime::otp::erlang;

pub fn closure(process: &Process, output: Term) -> std::result::Result<Term, Alloc> {
    process.anonymous_closure_with_env_from_slice(
        super::module(),
        0,
        Default::default(),
        Default::default(),
        2,
        Some(code),
        process.pid().into(),
        &[output],
    )
}

// Private

/// ```elixir
/// pushed from environment: (output)
/// pushed from arguments: (element, send_to)
/// pushed to stack: (output, element, send_to)
/// returned from call: N/A
/// full stack: (output, element, send_to)
/// fn (_, send_to) ->
///   spawn(Chain, :counter, [send_to, output])
/// end
/// ```
fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    // from environment
    let output = arc_process.stack_pop().unwrap();
    assert!(output.is_function());
    // from arguments
    let element = arc_process.stack_pop().unwrap();
    assert!(element.is_integer());
    let send_to = arc_process.stack_pop().unwrap();
    assert!(send_to.is_pid());

    // In `lumen` compiled code the compile would optimize this to a direct call of
    // `Scheduler::spawn(arc_process, module, function, arguments, counter_0_code)`, but we want
    // to demonstrate the the `lumen_runtime::code::set_apply_fn` system works here.

    let module = atom_unchecked("Elixir.Chain");
    let function = atom_unchecked("counter");
    let arguments = arc_process.list_from_slice(&[send_to, output]).unwrap();
    erlang::spawn_3::place_frame_with_arguments(
        arc_process,
        Placement::Replace,
        module,
        function,
        arguments,
    )
    .unwrap();

    Process::call_code(arc_process)
}
