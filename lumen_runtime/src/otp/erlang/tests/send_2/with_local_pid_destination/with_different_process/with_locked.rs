use super::*;

#[test]
fn with_atom_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::next_local_reference(process)
    });
}

#[test]
fn with_local_reference_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::next_local_reference(process)
    });
}

#[test]
fn with_empty_list_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::external_pid(1, 2, 3, &process).unwrap()
    });
}

#[test]
fn with_tuple_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::slice_to_tuple(&[], &process)
    });
}

#[test]
fn with_map_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::slice_to_map(&[], &process)
    });
}

#[test]
fn with_heap_binary_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| {
        Term::slice_to_binary(&[], &process)
    });
}

#[test]
fn with_subbinary_adds_heap_message_to_mailbox_and_returns_message() {
    with_adds_heap_message_to_mailbox_and_returns_message(|process| bitstring!(1 :: 1, &process));
}

fn with_adds_heap_message_to_mailbox_and_returns_message<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let different_process = process::local::new();
        let destination = different_process.pid;

        let _different_process_heap_lock = different_process.heap.lock().unwrap();

        let message = message(process);

        assert_eq!(erlang::send_2(destination, message, process), Ok(message));

        assert!(different_process
            .mailbox
            .lock()
            .unwrap()
            .iter()
            .any(|mailbox_message| match mailbox_message {
                Message::Heap {
                    message: heap_message,
                    ..
                } => heap_message == &message,
                _ => false,
            }))
    })
}
