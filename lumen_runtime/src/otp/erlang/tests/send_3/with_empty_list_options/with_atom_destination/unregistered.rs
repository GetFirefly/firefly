use super::*;

#[test]
fn with_atom_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_local_reference_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_message_errors_badarg() {
    with_message_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_message_errors_badarg() {
    with_message_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_message_errors_badarg() {
    with_message_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_message_errors_badarg() {
    with_message_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_message_errors_badarg() {
    with_message_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_local_pid_message_errors_badarg() {
    with_message_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_message_errors_badarg() {
    with_message_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_message_errors_badarg() {
    with_message_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_message_errors_badarg<M>(message: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let destination = registered_name();
        let message = message(process);
        let options = Term::EMPTY_LIST;

        assert_badarg!(erlang::send_3(destination, message, options, process));
    })
}
