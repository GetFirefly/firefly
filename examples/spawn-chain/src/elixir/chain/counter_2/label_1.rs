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
use std::convert::TryInto;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_2;

#[native_implemented::label]
fn result(process: &Process, ok: Term, next_pid: Term, output: Term) -> exception::Result<Term> {
    assert_eq!(ok, atom!("ok"));
    assert!(next_pid.is_pid());
    let _: Boxed<Closure> = output.try_into().unwrap();

    // Because there is a guardless match in the receive block, the first message will always be
    // removed and no loop is necessary.
    //
    // CANNOT be in `match` as it will hold temporaries in `match` arms causing a `park`.
    let received = process.mailbox.lock().borrow_mut().receive(process);

    match received {
        Some(Ok(n)) => {
            process.queue_frame_with_arguments(
                label_2::frame().with_arguments(false, &[n, next_pid, output]),
            );

            Ok(Term::NONE)
        }
        None => {
            process.wait();

            process
                .queue_frame_with_arguments(frame().with_arguments(false, &[ok, next_pid, output]));

            Ok(Term::NONE)
        }
        Some(Err(alloc_err)) => Err(alloc_err.into()),
    }
}
