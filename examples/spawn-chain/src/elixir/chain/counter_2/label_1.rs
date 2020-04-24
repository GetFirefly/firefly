use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::{Boxed, Closure, Encoded, Term};

use liblumen_otp::erlang;

use crate::elixir::chain::counter_2::label_2;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    next_pid: Term,
    output: Term,
) -> frames::Result {
    assert!(next_pid.is_pid());
    assert!(output.is_boxed_function());
    process.stack_push(output)?;
    process.stack_push(next_pid)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

/// ```elixir
/// # label 1
/// # pushed to stack: (next_pid, output)
/// # returned from called: :ok
/// # full stack: (:ok, next_pid, output)
/// # returns: :ok
/// receive do
///   n ->
///     output.("received #{n}")
///     sent = send(next_pid, n + 1)
///     output.("sent #{sent} to #{next_pid}")
/// end
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    // Because there is a guardless match in the receive block, the first message will always be
    // removed and no loop is necessary.
    //
    // CANNOT be in `match` as it will hold temporaries in `match` arms causing a `park`.
    let received = arc_process.mailbox.lock().borrow_mut().receive(arc_process);

    match received {
        Some(Ok(n)) => {
            let ok = arc_process.stack_pop().unwrap();
            assert!(ok.is_atom());
            // popped after receive in case of `Alloc` so stack is preserved
            let next_pid = arc_process.stack_pop().unwrap();
            assert!(next_pid.is_pid());
            let output = arc_process.stack_pop().unwrap();
            let _: Boxed<Closure> = output.try_into().unwrap();

            // ```elixir
            // # label 2
            // # pushed stack: (next_pid, output)
            // # returned from call: sum
            // # full stack: (sum, next_pid, output)
            // # returns: sent
            // sent = send(next_pid, sum)
            // output.("send #{sent} to #{next_pid}")
            // ```
            label_2::place_frame_with_arguments(arc_process, Placement::Replace, next_pid, output)
                .unwrap();

            // ```elixir
            // # pushed to stack: (n)
            // # return from call: N/A
            // # full stack: (n)
            // # returns: sum
            // n + 1
            let one = arc_process.integer(1).unwrap();
            erlang::add_2::place_frame_with_arguments(arc_process, Placement::Push, n, one)
                .unwrap();

            Process::call_native_or_yield(arc_process)
        }
        None => {
            Arc::clone(arc_process).wait();

            Ok(())
        }
        Some(Err(alloc_err)) => Err(alloc_err.into()),
    }
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}
