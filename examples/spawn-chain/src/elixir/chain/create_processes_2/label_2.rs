use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::code::stack::frame::Placement;
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use super::{frame, label_3};

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    output: Term,
) -> Result<(), Alloc> {
    process.stack_push(output)?;
    process.place_frame(frame(LOCATION, code), placement);

    Ok(())
}

/// ```elixir
/// # label 2
/// # pushed to stack: (output)
/// # returned from call: sent
/// # full stack: (sent, output)
/// # returns: :ok
/// receive do # and wait for the result to come back to us
///   final_answer when is_integer(final_answer) ->
///     output.("Result is #{inspect(final_answer)}")
///     final_answer
/// end
/// ```
#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    // locked mailbox scope
    let received = {
        let mailbox_guard = arc_process.mailbox.lock();
        let mut mailbox = mailbox_guard.borrow_mut();
        let seen = mailbox.seen();
        let mut found_position = None;

        for (position, message) in mailbox.iter().enumerate() {
            if seen < (position as isize) {
                let message_data = match message {
                    Message::Process(message::Process { data }) => data,
                    Message::HeapFragment(message::HeapFragment { data, .. }) => data,
                };

                if message_data.is_integer() {
                    let final_answer = match message {
                        Message::Process(message::Process { data }) => *data,
                        Message::HeapFragment(message::HeapFragment { data, .. }) => {
                            match data.clone_to_heap(&mut arc_process.acquire_heap()) {
                                Ok(heap_data) => heap_data,
                                Err(alloc) => {
                                    arc_process.reduce();

                                    return Err(alloc.into());
                                }
                            }
                        }
                    };

                    let sent = arc_process.stack_pop().unwrap();
                    assert!(sent.is_integer());
                    let output = arc_process.stack_pop().unwrap();
                    assert!(output.is_boxed_function());

                    label_3::place_frame_with_arguments(
                        arc_process,
                        Placement::Replace,
                        final_answer,
                    )
                    .unwrap();

                    let output_closure: Boxed<Closure> = output.try_into().unwrap();
                    // TODO use `<>` and `to_string` to more closely emulate interpolation
                    let binary = arc_process
                        .binary_from_str(&format!("Result is {}", final_answer))
                        .unwrap();
                    output_closure
                        .place_frame_with_arguments(arc_process, Placement::Push, vec![binary])
                        .unwrap();

                    found_position = Some(position);

                    break;
                } else {
                    // NOT in original Elixir source and would not be in compiled code, but helps
                    // debug runtime bugs leading to deadlocks.
                    panic!(
                        "Non-integer message ({:?}) received in {:?}",
                        message_data, arc_process
                    );
                }
            }
        }

        // separate because can't remove during iteration
        match found_position {
            Some(position) => {
                mailbox.remove(position, arc_process);
                mailbox.unmark_seen();

                true
            }
            None => {
                mailbox.mark_seen();

                false
            }
        }
    };

    arc_process.reduce();

    if received {
        Process::call_code(arc_process)
    } else {
        arc_process.wait();

        Ok(())
    }
}
