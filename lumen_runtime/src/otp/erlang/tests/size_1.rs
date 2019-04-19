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
fn with_tuple_without_elements_is_zero() {
    with_process(|process| {
        let empty_tuple_term = Term::slice_to_tuple(&[], &process);
        let zero_term = 0usize.into_process(&process);

        assert_eq!(erlang::size_1(empty_tuple_term, &process), Ok(zero_term));
    });
}

#[test]
fn with_tuple_with_elements_is_element_count() {
    with_process(|process| {
        let element_vec: Vec<Term> = (0..=2usize).map(|i| i.into_process(&process)).collect();
        let element_slice: &[Term] = element_vec.as_slice();
        let tuple_term = Term::slice_to_tuple(element_slice, &process);
        let arity_term = 3usize.into_process(&process);

        assert_eq!(erlang::size_1(tuple_term, &process), Ok(arity_term));
    });
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_is_byte_count() {
    with_process(|process| {
        let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &process);
        let byte_count_term = 3usize.into_process(&process);

        assert_eq!(
            erlang::size_1(heap_binary_term, &process),
            Ok(byte_count_term)
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count() {
    with_process(|process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &process);
        let byte_count_term = 2usize.into_process(&process);

        assert_eq!(
            erlang::size_1(subbinary_term, &process),
            Ok(byte_count_term)
        );
    });
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    with_process(|process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 0, &process);
        let byte_count_term = 2usize.into_process(&process);

        assert_eq!(
            erlang::size_1(subbinary_term, &process),
            Ok(byte_count_term)
        );
    });
}

fn errors_badarg<F>(binary_or_tuple: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::size_1(binary_or_tuple(&process), &process));
}
