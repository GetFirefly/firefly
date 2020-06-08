//! ```elixir
//! # label 2
//! # pushed to stack: (output)
//! # returned from call: sent
//! # full stack: (sent, output)
//! # returns: :ok
//! receive do # and wait for the result to come back to us
//!   final_answer when is_integer(final_answer) ->
//!     output.("Result is #{inspect(final_answer)}")
//!     final_answer
//! end
//! ```

use std::convert::TryInto;

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::message::{self, Message};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use super::label_3;

#[native_implemented::label]
fn result(process: &Process, sent: Term, output: Term) -> exception::Result<Term> {
    assert!(sent.is_integer());
    assert!(output.is_boxed_function());

    // locked mailbox scope
    let received = {
        let mailbox_guard = process.mailbox.lock();
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
                            match data.clone_to_heap(&mut process.acquire_heap()) {
                                Ok(heap_data) => heap_data,
                                Err(alloc) => {
                                    return Err(alloc.into());
                                }
                            }
                        }
                    };

                    let output_closure: Boxed<Closure> = output.try_into().unwrap();
                    // TODO use `<>` and `to_string` to more closely emulate interpolation
                    let binary = process
                        .binary_from_str(&format!("Result is {}", final_answer))
                        .unwrap();
                    process.queue_frame_with_arguments(
                        output_closure.frame_with_arguments(false, vec![binary]),
                    );

                    process.queue_frame_with_arguments(
                        label_3::frame().with_arguments(true, &[final_answer]),
                    );

                    found_position = Some(position);

                    break;
                } else {
                    // NOT in original Elixir source and would not be in compiled code, but helps
                    // debug runtime bugs leading to deadlocks.
                    panic!(
                        "Non-integer message ({:?}) received in {:?}",
                        message_data, process
                    );
                }
            }
        }

        // separate because can't remove during iteration
        match found_position {
            Some(position) => {
                mailbox.remove(position, process);
                mailbox.unmark_seen();

                true
            }
            None => {
                mailbox.mark_seen();

                false
            }
        }
    };

    if !received {
        process.wait();

        process.queue_frame_with_arguments(frame().with_arguments(false, &[sent, output]));
    }

    Ok(Term::NONE)
}
