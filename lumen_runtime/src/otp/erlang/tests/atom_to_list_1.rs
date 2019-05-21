use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_returns_chars_in_list() {
    with_process(|process| {
        let atom_name = "😈🤘";
        let atom_term = Term::str_to_atom(atom_name, DoNotCare).unwrap();

        assert_eq!(
            erlang::atom_to_list_1(atom_term, &process),
            Ok(Term::cons(
                128520.into_process(&process),
                Term::cons(129304.into_process(&process), Term::EMPTY_LIST, &process),
                &process
            ))
        );
    });
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0usize.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| {
        Term::slice_to_tuple(
            &[0.into_process(&process), 1.into_process(&process)],
            &process,
        )
    });
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn errors_badarg<F>(atom: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::atom_to_list_1(atom(&process), &process));
}
