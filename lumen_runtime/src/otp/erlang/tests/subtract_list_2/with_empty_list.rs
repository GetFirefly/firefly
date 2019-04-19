use super::*;

use num_traits::Num;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("term", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_returns_empty_list() {
    with_process(|process| {
        let minuend = Term::EMPTY_LIST;
        let subtrahend = Term::EMPTY_LIST;

        assert_eq!(
            erlang::subtract_list_2(minuend, subtrahend, &process),
            Ok(Term::EMPTY_LIST)
        );
    });
}

#[test]
fn with_list_returns_empty_list() {
    with_process(|process| {
        let minuend = Term::EMPTY_LIST;
        let subtrahend = Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process);

        assert_eq!(
            erlang::subtract_list_2(minuend, subtrahend, &process),
            Ok(Term::EMPTY_LIST)
        );
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(2.into_process(&process), 3.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 1.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| {
        <BigInt as Num>::from_str_radix("576460752303423490", 10)
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
    errors_badarg(|_| Term::local_pid(1, 2).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(4, 5, 6, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_errors_badarg() {
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
        Term::subbinary(original, 0, 7, 2, 0, &process)
    });
}

fn errors_badarg<F>(subtrahend: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        let minuend = Term::EMPTY_LIST;
        let subtrahend = subtrahend(&process);

        (minuend, subtrahend)
    });
}
