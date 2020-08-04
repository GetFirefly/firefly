//! ```elixir
//! pushed from environment: (output)
//! pushed from arguments: (element, send_to)
//! pushed to stack: (output, element, send_to)
//! returned from call: N/A
//! full stack: (output, element, send_to)
//! fn (_, send_to) ->
//!   spawn(Chain, :counter, [send_to, output])
//! end
//! ```

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

pub fn closure(process: &Process, output: Term) -> std::result::Result<Term, Alloc> {
    process.anonymous_closure_with_env_from_slice(
        super::module(),
        0,
        Default::default(),
        Default::default(),
        2,
        CLOSURE_NATIVE,
        process.pid().into(),
        &[output],
    )
}

// Private

#[native_implemented::function(Elixir.Chain:create_processes_reducer/3)]
fn result(
    process: &Process,
    element: Term,
    send_to: Term,
    output: Term,
) -> exception::Result<Term> {
    // from arguments
    assert!(element.is_integer());
    assert!(send_to.is_pid());
    // from environment
    assert!(
        output.is_boxed_function(),
        "Output ({:?}) is not a function",
        output
    );

    // In `lumen` compiled code the compile would optimize this to a direct call of
    // `Scheduler::spawn(arc_process, module, function, arguments, counter_0_code)`, but we want
    // to demonstrate the the `lumen_rt_full::code::set_apply_fn` system works here.

    let module = Atom::str_to_term("Elixir.Chain");
    let function = Atom::str_to_term("counter");
    let arguments = process.list_from_slice(&[send_to, output])?;

    process.queue_frame_with_arguments(
        erlang::spawn_3::frame().with_arguments(false, &[module, function, arguments]),
    );

    Ok(Term::NONE)
}
