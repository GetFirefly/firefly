use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_is_false() {
    is_reference(|_| Term::str_to_atom("atom", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_is_true() {
    is_reference(|process| Term::next_local_reference(process), true);
}

#[test]
fn with_empty_list_is_false() {
    is_reference(|_| Term::EMPTY_LIST, false);
}

#[test]
fn with_list_is_false() {
    is_reference(|process| list_term(&process), false);
}

#[test]
fn with_small_integer_is_false() {
    is_reference(|process| 0.into_process(&process), false);
}

#[test]
fn with_big_integer_is_false() {
    is_reference(
        |process| (integer::small::MAX + 1).into_process(&process),
        false,
    );
}

#[test]
fn with_float_is_false() {
    is_reference(|process| 1.0.into_process(&process), false);
}

#[test]
fn with_local_pid_is_false() {
    is_reference(|_| Term::local_pid(0, 0).unwrap(), false);
}

#[test]
fn with_external_pid_is_false() {
    is_reference(
        |process| Term::external_pid(1, 0, 0, &process).unwrap(),
        false,
    );
}

#[test]
fn with_tuple_is_false() {
    is_reference(|process| Term::slice_to_tuple(&[], &process), false);
}

#[test]
fn with_map_is_false() {
    is_reference(|process| Term::slice_to_map(&[], &process), false);
}

#[test]
fn with_heap_binary_is_false() {
    is_reference(|process| Term::slice_to_binary(&[], &process), false);
}

#[test]
fn with_subbinary_is_false() {
    is_reference(|process| bitstring!(1 :: 1, &process), false);
}

fn is_reference<T>(term: T, expected: bool)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        assert_eq!(erlang::is_reference_1(term(&process)), expected.into());
    });
}
