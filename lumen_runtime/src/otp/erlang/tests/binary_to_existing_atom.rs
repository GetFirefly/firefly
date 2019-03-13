use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(atom_term, encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(Term::EMPTY_LIST, encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(list_term, encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term = 0usize.into_process(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(small_integer_term, encoding_term, &mut process),
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
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(big_integer_term, encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);
    let encoding_term = Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(float_term, encoding_term, &mut process),
        Err(bad_argument!()),
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

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(tuple_term, encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_without_encoding_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(
            heap_binary_term,
            0.into_process(&mut process),
            &mut process
        ),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_with_invalid_encoding_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let invalid_encoding_term =
        Term::str_to_atom("invalid_encoding", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, invalid_encoding_term, &mut process),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_heap_binary_with_valid_encoding_without_existing_atom_returns_atom() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("😈".as_bytes(), &mut process);
    let latin1_atom_term = Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, latin1_atom_term, &mut process),
        Err(bad_argument!()),
        &mut process
    );
    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, unicode_atom_term, &mut process),
        Err(bad_argument!()),
        &mut process
    );
    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, utf8_atom_term, &mut process),
        Err(bad_argument!()),
        &mut process
    );
}

#[test]
fn with_heap_binary_with_valid_encoding_with_existing_atom_returns_atom() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary("😈".as_bytes(), &mut process);
    let latin1_atom_term = Term::str_to_atom("latin1", Existence::DoNotCare, &mut process).unwrap();
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();
    let atom_term = Term::str_to_atom("😈", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, latin1_atom_term, &mut process),
        Ok(atom_term),
        &mut process
    );
    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, unicode_atom_term, &mut process),
        Ok(atom_term),
        &mut process
    );
    assert_eq_in_process!(
        erlang::binary_to_existing_atom(heap_binary_term, utf8_atom_term, &mut process),
        Ok(atom_term),
        &mut process
    );
}

#[test]
fn with_subbinary_with_bit_count_returns_bad_argument() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
        Err(bad_argument!()),
        &mut process
    )
}

#[test]
fn with_subbinary_without_bit_count_without_existing_atom_returns_bad_argument() {
    let mut process: Process = Default::default();
    let binary_term = Term::slice_to_binary("😈🤘".as_bytes(), &mut process);
    let subbinary_term = Term::subbinary(binary_term, 4, 0, 4, 0, &mut process);
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
        Err(bad_argument!()),
        &mut process
    )
}

#[test]
fn with_subbinary_without_bit_count_with_existing_atom_returns_atom_with_bytes() {
    let mut process: Process = Default::default();
    let binary_term = Term::slice_to_binary("😈🤘".as_bytes(), &mut process);
    let subbinary_term = Term::subbinary(binary_term, 4, 0, 4, 0, &mut process);
    let unicode_atom_term =
        Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
    let atom_term = Term::str_to_atom("🤘", Existence::DoNotCare, &mut process);

    assert_eq_in_process!(
        erlang::binary_to_existing_atom(subbinary_term, unicode_atom_term, &mut process),
        atom_term,
        &mut process
    )
}
