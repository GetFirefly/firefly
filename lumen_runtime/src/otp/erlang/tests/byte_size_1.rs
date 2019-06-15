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
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
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
fn with_heap_binary_is_byte_count() {
    with_process(|process| {
        let heap_binary_term = Term::slice_to_binary(&[1], &process);

        assert_eq!(
            erlang::byte_size_1(heap_binary_term, &process),
            Ok(1.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    with_process(|process| {
        let binary_term = Term::slice_to_binary(&[0, 1], &process);
        let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &process);

        assert_eq!(
            erlang::byte_size_1(subbinary_term, &process),
            Ok(1.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count_plus_one() {
    with_process(|process| {
        let binary_term = Term::slice_to_binary(&[0, 1, 0b0100_0000], &process);
        let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 3, &process);

        assert_eq!(
            erlang::byte_size_1(subbinary_term, &process),
            Ok(2.into_process(&process))
        );
    });
}

fn errors_badarg<F>(bit_string: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::byte_size_1(bit_string(&process), &process));
}
