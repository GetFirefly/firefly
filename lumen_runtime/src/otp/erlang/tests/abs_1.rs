use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[0], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
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
fn with_small_integer_that_is_negative_returns_positive() {
    with_process(|process| {
        let negative: isize = -1;
        let negative_term = negative.into_process(&process);

        let positive = -negative;
        let positive_term = positive.into_process(&process);

        assert_eq!(erlang::abs_1(negative_term, &process), Ok(positive_term));
    });
}

#[test]
fn with_small_integer_that_is_positive_returns_self() {
    with_process(|process| {
        let positive_term = 1usize.into_process(&process);

        assert_eq!(erlang::abs_1(positive_term, &process), Ok(positive_term));
    });
}

#[test]
fn with_big_integer_that_is_negative_returns_positive() {
    with_process(|process| {
        let negative: isize = small::MIN - 1;
        let negative_term = negative.into_process(&process);

        assert_eq!(negative_term.tag(), Boxed);

        let unboxed_negative_term: &Term = negative_term.unbox_reference();

        assert_eq!(unboxed_negative_term.tag(), BigInteger);

        let positive = -negative;
        let positive_term = positive.into_process(&process);

        assert_eq!(erlang::abs_1(negative_term, &process), Ok(positive_term));
    });
}

#[test]
fn with_big_integer_that_is_positive_return_self() {
    with_process(|process| {
        let positive_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process);

        assert_eq!(positive_term.tag(), Boxed);

        let unboxed_positive_term: &Term = positive_term.unbox_reference();

        assert_eq!(unboxed_positive_term.tag(), BigInteger);

        assert_eq!(erlang::abs_1(positive_term, &process), Ok(positive_term));
    });
}

#[test]
fn with_float_that_is_negative_returns_positive() {
    with_process(|process| {
        let negative = -1.0;
        let negative_term = negative.into_process(&process);

        assert_eq!(negative_term.tag(), Boxed);

        let unboxed_negative_term: &Term = negative_term.unbox_reference();

        assert_eq!(unboxed_negative_term.tag(), Float);

        let positive = -negative;
        let positive_term = positive.into_process(&process);

        assert_eq!(erlang::abs_1(negative_term, &process), Ok(positive_term));
    });
}

#[test]
fn with_float_that_is_positive_return_self() {
    with_process(|process| {
        let positive_term: Term = 1.0.into_process(&process);

        assert_eq!(positive_term.tag(), Boxed);

        let unboxed_positive_term: &Term = positive_term.unbox_reference();

        assert_eq!(unboxed_positive_term.tag(), Float);

        assert_eq!(erlang::abs_1(positive_term, &process), Ok(positive_term));
    });
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

fn errors_badarg<F>(number: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::abs_1(number(&process), &process));
}
