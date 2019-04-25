use super::*;

#[test]
fn with_atom_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::local_reference(&process));
}

#[test]
fn with_local_reference_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_message_returns_noconnect() {
    with_message_returns_noconnect(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_returns_noconnect() {
    with_message_returns_noconnect(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_message_returns_noconnect() {
    with_message_returns_noconnect(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_message_returns_noconnect() {
    with_message_returns_noconnect(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_message_returns_noconnect() {
    with_message_returns_noconnect(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_message_returns_noconnect() {
    with_message_returns_noconnect(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_message_returns_noconnect() {
    with_message_returns_noconnect(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_message_returns_noconnect() {
    with_message_returns_noconnect(|process| bitstring!(1 :: 1, &process));
}

fn with_message_returns_noconnect<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process_arc(|process_arc| {
        let name = registered_name();

        assert_eq!(
            erlang::register_2(name, process_arc.pid, process_arc.clone()),
            Ok(true.into())
        );

        let destination = Term::slice_to_tuple(
            &[
                name,
                Term::str_to_atom("node@example.com", DoNotCare).unwrap(),
            ],
            &process_arc,
        );
        let message = message(&process_arc);
        let options = options(&process_arc);

        assert_eq!(
            erlang::send_3(destination, message, options, &process_arc),
            Ok(Term::str_to_atom("noconnect", DoNotCare).unwrap())
        );
    })
}
