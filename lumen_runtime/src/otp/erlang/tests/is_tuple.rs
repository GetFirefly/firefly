use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_is_false() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(atom_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_empty_list_is_false() {
    let mut process: Process = Default::default();
    let empty_list_term = Term::EMPTY_LIST;
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(empty_list_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_list_is_false() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(list_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_small_integer_is_false() {
    let mut process: Process = Default::default();
    let small_integer_term = 0.into_process(&mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(small_integer_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_big_integer_is_false() {
    let mut process: Process = Default::default();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(big_integer_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_float_is_false() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(float_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_local_pid_is_true() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(local_pid_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_external_pid_is_true() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(external_pid_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_tuple_is_true() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);
    let true_term = true.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(tuple_term, &mut process),
        true_term,
        process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(heap_binary_term, &mut process),
        false_term,
        process
    );
}

#[test]
fn with_subbinary_is_false() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let false_term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_tuple(subbinary_term, &mut process),
        false_term,
        process
    );
}
