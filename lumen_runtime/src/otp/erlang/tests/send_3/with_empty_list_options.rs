use super::*;

mod with_atom_destination;
mod with_local_pid_destination;
mod with_tuple_destination;

#[test]
fn with_local_reference_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_destination_errors_badarg() {
    with_destination_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_destination_errors_badarg() {
    with_destination_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_destination_errors_badarg() {
    with_destination_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_destination_errors_badarg() {
    with_destination_errors_badarg(|process| {
        (crate::integer::small::MAX + 1).into_process(&process)
    });
}

#[test]
fn with_float_destination_errors_badarg() {
    with_destination_errors_badarg(|process| 0.0.into_process(&process));
}

#[test]
fn with_external_pid_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_map_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_destination_errors_badarg() {
    with_destination_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_destination_errors_badarg() {
    with_destination_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_destination_errors_badarg<D>(destination: D)
where
    D: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let message = Term::str_to_atom("message", DoNotCare).unwrap();
        let options = Term::EMPTY_LIST;

        assert_badarg!(erlang::send_3(
            destination(process),
            message,
            options,
            process
        ));
    });
}
