use super::*;

#[test]
fn with_atom_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::next_local_reference(process));
}

#[test]
fn with_local_reference_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_message_panics_unimplemented() {
    with_message_panics_unimplemented(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_message_panics_unimplemented() {
    with_message_panics_unimplemented(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_message_panics_unimplemented() {
    with_message_panics_unimplemented(|process| bitstring!(1 :: 1, &process));
}

fn with_message_panics_unimplemented<M>(message: M)
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
        let options = Term::EMPTY_LIST;

        let result = std::panic::catch_unwind(|| {
            erlang::send_3(destination, message, options, &process_arc)
        });

        assert!(result.is_err());

        let err = result.unwrap_err();

        assert_eq!(
            err.downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().map(|e| *e))
                .unwrap(),
            "not yet implemented: distribution"
        );
    })
}
