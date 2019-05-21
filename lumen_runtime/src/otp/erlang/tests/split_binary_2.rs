use super::*;

use crate::integer;
use crate::process::IntoProcess;

mod with_heap_binary;
mod with_subbinary;

#[test]
fn with_atom_errors_badarg() {
    with_binary_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    with_binary_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarg() {
    with_binary_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    with_binary_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_badarg() {
    with_binary_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    with_binary_errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    with_binary_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    with_binary_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    with_binary_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    with_binary_errors_badarg(|process| {
        Term::slice_to_tuple(
            &[0.into_process(&process), 1.into_process(&process)],
            &process,
        )
    });
}

#[test]
fn with_map_errors_badarg() {
    with_binary_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

fn with_binary_errors_badarg<F>(binary: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let position = 0.into_process(&process);

        erlang::split_binary_2(binary(&process), position, &process)
    });
}
