use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_atom_reason() {
    errors(|_| Term::str_to_atom("reason", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_local_reference() {
    errors(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_errors_empty_list_reason() {
    errors(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_list_reason() {
    errors(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_small_integer_reason() {
    errors(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_big_integer_reason() {
    errors(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_float_reason() {
    errors(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_local_pid_reason() {
    errors(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_external_pid_reason() {
    errors(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_tuple_reason() {
    errors(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_map_reason() {
    errors(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_heap_binary_reason() {
    errors(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_subbinary_reason() {
    errors(|process| bitstring!(1 :: 1, &process))
}

fn errors<R>(reason: R)
where
    R: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let reason = reason(&process);

        assert_error!(erlang::error_1(reason), reason);
    });
}
