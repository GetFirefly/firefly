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
    errors_badarg(|mut process| {
        let minuend = Term::local_reference(&mut process);
        let subtrahend = Term::local_reference(&mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        );
        let subtrahend = Term::cons(
            2.into_process(&mut process),
            3.into_process(&mut process),
            &mut process,
        );

        (minuend, subtrahend)
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = 0.into_process(&mut process);
        let subtrahend = 1.into_process(&mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process);
        let subtrahend = <BigInt as Num>::from_str_radix("576460752303423490", 10)
            .unwrap()
            .into_process(&mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = 1.0.into_process(&mut process);
        let subtrahend = 2.0.into_process(&mut process);

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
    errors_badarg(|mut process| {
        let minuend = Term::external_pid(1, 2, 3, &mut process).unwrap();
        let subtrahend = Term::external_pid(4, 5, 6, &mut process).unwrap();

        (minuend, subtrahend)
    });
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = Term::slice_to_tuple(&[], &mut process);
        let subtrahend = Term::slice_to_tuple(&[], &mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_map_is_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = Term::slice_to_map(&[], &mut process);
        let subtrahend = Term::slice_to_map(&[], &mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| {
        let minuend = Term::slice_to_binary(&[], &mut process);
        let subtrahend = Term::slice_to_binary(&[], &mut process);

        (minuend, subtrahend)
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|mut process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        let minuend = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
        let subtrahend = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);

        (minuend, subtrahend)
    });
}

fn errors_badarg<F>(minuend_subtrahend: F)
where
    F: FnOnce(&mut Process) -> (Term, Term),
{
    super::errors_badarg(|mut process| {
        let (minuend, subtrahend) = minuend_subtrahend(&mut process);

        erlang::subtract_list_2(minuend, subtrahend, &mut process)
    });
}
