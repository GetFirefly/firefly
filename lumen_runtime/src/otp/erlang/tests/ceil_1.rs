use super::*;

use num_traits::Num;

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
fn with_small_integer_returns_same() {
    with_process(|process| {
        let small_integer_term: Term = 0.into_process(&process);

        let result = erlang::ceil_1(small_integer_term, &process);

        assert_eq!(result, Ok(small_integer_term));
        assert_eq!(result.unwrap().tagged, small_integer_term.tagged);
    });
}

#[test]
fn with_big_integer_returns_same() {
    with_process(|process| {
        let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&process);

        let result = erlang::ceil_1(big_integer_term, &process);

        assert_eq!(result, Ok(big_integer_term));
        assert_eq!(result.unwrap().tagged, big_integer_term.tagged);
    });
}

#[test]
fn with_float_without_fraction_returns_integer() {
    with_process(|process| {
        let float_term = 1.0.into_process(&process);

        assert_eq!(
            erlang::ceil_1(float_term, &process),
            Ok(1.into_process(&process))
        );
    });
}

#[test]
fn with_float_with_fraction_rounds_up_to_next_integer() {
    with_process(|process| {
        let float_term = (-1.1).into_process(&process);

        let result = erlang::ceil_1(float_term, &process);

        assert_eq!(result, Ok((-1).into_process(&process)));
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

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[1], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0, 1], &process);
        Term::subbinary(original, 1, 0, 1, 0, &process)
    });
}

fn errors_badarg<F>(number: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::ceil_1(number(&process), &process));
}
