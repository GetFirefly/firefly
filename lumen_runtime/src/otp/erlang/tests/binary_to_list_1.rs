use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
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
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_returns_list_of_bytes() {
    with_process(|process| {
        let binary = Term::slice_to_binary(&[0, 1, 2], &process);

        assert_eq!(
            erlang::binary_to_list_1(binary, &process),
            Ok(Term::cons(
                0.into_process(&process),
                Term::cons(
                    1.into_process(&process),
                    Term::cons(2.into_process(&process), Term::EMPTY_LIST, &process),
                    &process
                ),
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_without_bit_count_returns_list_of_bytes() {
    with_process(|process| {
        // <<1::1, 0, 1, 2>>
        let original = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &process);
        let binary = Term::subbinary(original, 0, 1, 3, 0, &process);

        assert_eq!(
            erlang::binary_to_list_1(binary, &process),
            Ok(Term::cons(
                0.into_process(&process),
                Term::cons(
                    1.into_process(&process),
                    Term::cons(2.into_process(&process), Term::EMPTY_LIST, &process),
                    &process
                ),
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_errors_badarg() {
    errors_badarg(|process| {
        // <<1::1, 0, 1, 2>>
        let original = Term::slice_to_binary(&[128, 0, 129, 0b0000_0000], &process);
        Term::subbinary(original, 0, 0, 3, 1, &process)
    });
}

fn errors_badarg<F>(binary: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::binary_to_list_1(binary(&process), &process));
}
