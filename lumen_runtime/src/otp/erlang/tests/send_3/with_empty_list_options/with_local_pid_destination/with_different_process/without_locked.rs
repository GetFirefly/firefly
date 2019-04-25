use super::*;

#[test]
fn with_atom_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::local_reference(&process)
    });
}

#[test]
fn with_local_reference_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::local_reference(&process)
    });
}

#[test]
fn with_empty_list_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        0.0.into_process(&process)
    });
}

#[test]
fn with_local_pid_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::external_pid(1, 2, 3, &process).unwrap()
    });
}

#[test]
fn with_tuple_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_tuple(&[], &process)
    });
}

#[test]
fn with_map_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_map(&[], &process)
    });
}

#[test]
fn with_heap_binary_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_binary(&[], &process)
    });
}

#[test]
fn with_subbinary_message_adds_process_message_to_mailbox_and_returns_ok() {
    with_message_adds_process_message_to_mailbox_and_returns_ok(
        |process| bitstring!(1 :: 1, &process),
    );
}

fn with_message_adds_process_message_to_mailbox_and_returns_ok<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let different_process = process::local::new();
        let destination = different_process.pid;
        let message = message(process);
        let options = Term::EMPTY_LIST;

        assert_eq!(
            erlang::send_3(destination, message, options, process),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        assert!(different_process
            .mailbox
            .lock()
            .unwrap()
            .iter()
            .any(|mailbox_message| match mailbox_message {
                Message::Process(process_message) => process_message == &message,
                _ => false,
            }))
    })
}
