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
fn with_heap_binary_returns_list_of_integer() {
    with_process(|process| {
        let bit_string = Term::slice_to_binary(&[0], &process);

        assert_eq!(
            erlang::bitstring_to_list_1(bit_string, &process),
            Ok(Term::cons(
                0.into_process(&process),
                Term::EMPTY_LIST,
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_without_bit_count_returns_list_of_integer() {
    with_process(|process| {
        let original = Term::slice_to_binary(&[0, 1, 0b010], &process);
        let bit_string = Term::subbinary(original, 1, 0, 1, 0, &process);

        assert_eq!(
            erlang::bitstring_to_list_1(bit_string, &process),
            Ok(Term::cons(
                1.into_process(&process),
                Term::EMPTY_LIST,
                &process
            ))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_returns_list_of_integer_with_bitstring_for_bit_count() {
    with_process(|process| {
        let bitstring = bitstring!(0, 1, 0b010 :: 3, &process);

        assert_eq!(
            erlang::bitstring_to_list_1(bitstring, &process),
            Ok(Term::cons(
                0.into_process(&process),
                Term::cons(
                    1.into_process(&process),
                    bitstring!(0b010 :: 3, &process),
                    &process
                ),
                &process
            ))
        );
    });
}

fn errors_badarg<F>(bit_string: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::bitstring_to_list_1(bit_string(&process), &process));
}
