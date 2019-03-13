use super::*;

use num_traits::Num;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::length(atom_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_empty_list_is_zero() {
    let mut process: Process = Default::default();
    let zero_term = 0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::length(Term::EMPTY_LIST, &mut process),
        Ok(zero_term),
        process
    );
}

#[test]
fn with_improper_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
    let tail_term = Term::str_to_atom("tail", Existence::DoNotCare, &mut process).unwrap();
    let improper_list_term = Term::cons(head_term, tail_term, &mut process);

    assert_eq_in_process!(
        erlang::length(improper_list_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_list_is_length() {
    let mut process: Process = Default::default();
    let list_term = (0..=2).rfold(Term::EMPTY_LIST, |acc, i| {
        Term::cons(i.into_process(&mut process), acc, &mut process)
    });

    assert_eq_in_process!(
        erlang::length(list_term, &mut process),
        Ok(3.into_process(&mut process)),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term = 0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::length(small_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::length(big_integer_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::length(float_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_eq_in_process!(
        erlang::length(local_pid_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::length(external_pid_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::length(tuple_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_eq_in_process!(
        erlang::length(heap_binary_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_subbinary_is_false() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_eq_in_process!(
        erlang::length(subbinary_term, &mut process),
        Err(bad_argument!()),
        process
    );
}
