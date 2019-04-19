use super::*;

use num_traits::Num;

mod with_empty_list;
mod with_list;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| {
        let minuend = Term::str_to_atom("list", DoNotCare).unwrap();
        let subtrahend = Term::str_to_atom("term", DoNotCare).unwrap();

        (minuend, subtrahend)
    });
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::local_reference(&process);
        let subtrahend = Term::local_reference(&process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::cons(0.into_process(&process), 1.into_process(&process), &process);
        let subtrahend = Term::cons(2.into_process(&process), 3.into_process(&process), &process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| {
        let minuend = 0.into_process(&process);
        let subtrahend = 1.into_process(&process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| {
        let minuend = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process);
        let subtrahend = <BigInt as Num>::from_str_radix("576460752303423490", 10)
            .unwrap()
            .into_process(&process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| {
        let minuend = 1.0.into_process(&process);
        let subtrahend = 2.0.into_process(&process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| {
        let minuend = Term::local_pid(0, 1).unwrap();
        let subtrahend = Term::local_pid(1, 2).unwrap();

        (minuend, subtrahend)
    });
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::external_pid(1, 2, 3, &process).unwrap();
        let subtrahend = Term::external_pid(4, 5, 6, &process).unwrap();

        (minuend, subtrahend)
    });
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::slice_to_tuple(&[], &process);
        let subtrahend = Term::slice_to_tuple(&[], &process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_map_is_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::slice_to_map(&[], &process);
        let subtrahend = Term::slice_to_map(&[], &process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| {
        let minuend = Term::slice_to_binary(&[], &process);
        let subtrahend = Term::slice_to_binary(&[], &process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        let minuend = Term::subbinary(binary_term, 0, 7, 2, 1, &process);
        let subtrahend = Term::subbinary(binary_term, 0, 7, 2, 0, &process);

        (minuend, subtrahend)
    });
}

fn errors_badarg<F>(minuend_subtrahend: F)
where
    F: FnOnce(&Process) -> (Term, Term),
{
    super::errors_badarg(|process| {
        let (minuend, subtrahend) = minuend_subtrahend(&process);

        erlang::subtract_list_2(minuend, subtrahend, &process)
    });
}
