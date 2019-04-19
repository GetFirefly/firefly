use super::*;

use num_traits::Num;

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
fn with_heap_binary_is_eight_times_byte_count() {
    with_process(|process| {
        let heap_binary_term = Term::slice_to_binary(&[1], &process);

        assert_eq!(
            erlang::bit_size_1(heap_binary_term, &process),
            Ok(8.into_process(&process))
        );
    });
}

#[test]
fn with_subbinary_is_eight_times_byte_count_plus_bit_count() {
    with_process(|process| {
        let bitstring = bitstring!(0, 1, 0b010 :: 3, &process);

        assert_eq!(
            erlang::bit_size_1(bitstring, &process),
            Ok(19.into_process(&process))
        );
    });
}

fn errors_badarg<F>(bit_string: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::bit_size_1(bit_string(&process), &process));
}
