use super::*;

use crate::process::local::pid_to_process;
use crate::process::Status;

mod with_atom_module;

#[test]
fn with_small_integer_module_errors_badarg() {
    with_module_errors_badarg(|process| 0.into_process(process));
}

#[test]
fn with_float_module_errors_badarg() {
    with_module_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_big_integer_module_errors_badarg() {
    with_module_errors_badarg(|process| (integer::small::MAX + 1).into_process(process));
}

#[test]
fn with_local_reference_module_errors_badarg() {
    with_module_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_local_pid_module_errors_badarg() {
    with_module_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_module_errors_badarg() {
    with_module_errors_badarg(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_module_errors_badarg() {
    with_module_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_module_errors_badarg() {
    with_module_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_empty_list_module_errors_badarg() {
    with_module_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_module_errors_badarg() {
    with_module_errors_badarg(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_heap_binary_module_errors_badarg() {
    with_module_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_module_errors_badarg() {
    with_module_errors_badarg(|process| bitstring!(1 ::1, &process));
}

fn with_module_errors_badarg<M>(module: M)
where
    M: FnOnce(&Process) -> Term,
{
    errors_badarg(|process| {
        let function = Term::str_to_atom("function", DoNotCare).unwrap();
        let arguments = Term::EMPTY_LIST;

        erlang::spawn_3(module(process), function, arguments, process)
    })
}
