use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_is_false() {
    is_pid(|_| Term::str_to_atom("atom", DoNotCare).unwrap(), false);
}

#[test]
fn with_local_reference_is_false() {
    is_pid(|process| Term::next_local_reference(process), false);
}

#[test]
fn with_empty_list_is_false() {
    is_pid(|_| Term::EMPTY_LIST, false);
}

#[test]
fn with_list_is_false() {
    is_pid(
        |process| {
            let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
            Term::cons(head_term, Term::EMPTY_LIST, &process)
        },
        false,
    );
}

#[test]
fn with_small_integer_is_false() {
    is_pid(|process| 0.into_process(&process), false);
}

#[test]
fn with_big_integer_is_false() {
    is_pid(
        |process| (integer::small::MAX + 1).into_process(&process),
        false,
    );
}

#[test]
fn with_float_is_false() {
    is_pid(|process| 1.0.into_process(&process), false);
}

#[test]
fn with_local_pid_is_true() {
    is_pid(|_| Term::local_pid(0, 0).unwrap(), true);
}

#[test]
fn with_external_pid_is_true() {
    is_pid(
        |process| Term::external_pid(1, 0, 0, &process).unwrap(),
        true,
    );
}

#[test]
fn with_tuple_is_false() {
    is_pid(|process| Term::slice_to_tuple(&[], &process), false);
}

#[test]
fn with_map_is_false() {
    is_pid(|process| Term::slice_to_map(&[], &process), false);
}

#[test]
fn with_heap_binary_is_false() {
    is_pid(|process| Term::slice_to_binary(&[], &process), false);
}

#[test]
fn with_subbinary_is_false() {
    is_pid(|process| bitstring!(1 :: 1, &process), false);
}

fn is_pid<T>(term: T, expected: bool)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| assert_eq!(erlang::is_pid_1(term(&process)), expected.into()))
}
