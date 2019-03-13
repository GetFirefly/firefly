use super::*;

use num_traits::Num;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::byte_size(atom_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();

    assert_eq_in_process!(
        erlang::byte_size(Term::EMPTY_LIST, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);

    assert_eq_in_process!(
        erlang::byte_size(list_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term: Term = 0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::byte_size(small_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::byte_size(big_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::byte_size(float_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_eq_in_process!(
        erlang::byte_size(local_pid_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::byte_size(external_pid_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::byte_size(tuple_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_is_byte_count() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[1], &mut process);

    assert_eq_in_process!(
        erlang::byte_size(heap_binary_term, &mut process),
        Ok(1.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    let mut process: Process = Default::default();
    let binary_term = Term::slice_to_binary(&[0, 1], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);

    assert_eq_in_process!(
        erlang::byte_size(subbinary_term, &mut process),
        Ok(1.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count_plus_one() {
    let mut process: Process = Default::default();
    let binary_term = Term::slice_to_binary(&[0, 1, 0b0100_0000], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 3, &mut process);

    assert_eq_in_process!(
        erlang::byte_size(subbinary_term, &mut process),
        Ok(2.into_process(&mut process)),
        process
    );
}
