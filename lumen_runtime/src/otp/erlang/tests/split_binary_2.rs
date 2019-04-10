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
    with_binary_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    with_binary_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    with_binary_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_errors_badarg() {
    with_binary_errors_badarg(|mut process| 0.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    with_binary_errors_badarg(|mut process| (integer::small::MAX + 1).into_process(&mut process));
}

#[test]
fn with_float_errors_badarg() {
    with_binary_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_errors_badarg() {
    with_binary_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    with_binary_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    with_binary_errors_badarg(|mut process| {
        Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process,
        )
    });
}

#[test]
fn with_map_errors_badarg() {
    with_binary_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

fn with_binary_errors_badarg<F>(binary: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let position = 0.into_process(&mut process);

        erlang::split_binary_2(binary(&mut process), position, &mut process)
    });
}
