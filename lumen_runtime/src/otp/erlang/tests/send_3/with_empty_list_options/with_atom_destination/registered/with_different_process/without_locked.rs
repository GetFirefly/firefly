use super::*;

#[test]
fn with_atom_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| Term::local_reference(&process));
}

#[test]
fn with_local_reference_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::external_pid(1, 2, 3, &process).unwrap()
    });
}

#[test]
fn with_tuple_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_tuple(&[], &process)
    });
}

#[test]
fn with_map_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_map(&[], &process)
    });
}

#[test]
fn with_heap_binary_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| {
        Term::slice_to_binary(&[], &process)
    });
}

#[test]
fn with_subbinary_adds_process_message_to_mailbox_and_returns_ok() {
    with_adds_process_message_to_mailbox_and_returns_ok(|process| bitstring!(1 :: 1, &process));
}

fn with_adds_process_message_to_mailbox_and_returns_ok<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let different_process = process::local::new();
        let destination = registered_name();

        assert_eq!(
            erlang::register_2(destination, different_process.pid, process_arc.clone()),
            Ok(true.into())
        );

        let message = message(&process_arc);
        let options = Term::EMPTY_LIST;

        assert_eq!(
            erlang::send_3(destination, message, options, &process_arc),
            Ok(Term::str_to_atom("ok", DoNotCare).unwrap())
        );

        assert!(has_process_message(&different_process, message));
    })
}
