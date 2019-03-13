use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_without_encoding_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(atom_term, 0.into_process(&mut process), &mut process),
        process
    );
}

#[test]
fn with_atom_with_invalid_encoding_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
    let invalid_encoding_atom_term =
        Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(atom_term, invalid_encoding_atom_term, &mut process),
        process
    );
}

#[test]
fn with_atom_with_encoding_atom_returns_name_in_binary() {
    let mut process: Process = Default::default();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, Existence::DoNotCare, &mut process).unwrap();
    let latin1_atom_term = Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::atom_to_binary(atom_term, latin1_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
        process
    );
    assert_eq_in_process!(
        erlang::atom_to_binary(atom_term, unicode_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
        process
    );
    assert_eq_in_process!(
        erlang::atom_to_binary(atom_term, utf8_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(Term::EMPTY_LIST, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(list_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term = 0usize.into_process(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(small_integer_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(big_integer_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(float_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(local_pid_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(external_pid_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 1.into_process(&mut process)],
        &mut process,
    );
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(tuple_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_heap_binary_is_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(heap_binary_term, encoding_term, &mut process),
        process
    );
}

#[test]
fn with_subbinary_is_bad_argument() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::atom_to_binary(subbinary_term, encoding_term, &mut process),
        process
    );
}
