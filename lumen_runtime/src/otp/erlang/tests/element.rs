use super::*;

use num_traits::Num;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::element(atom_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();

    assert_eq_in_process!(
        erlang::element(Term::EMPTY_LIST, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);

    assert_eq_in_process!(
        erlang::element(list_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term: Term = 0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::element(small_integer_term, 0.into_process(&mut process)),
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
        erlang::element(big_integer_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::element(float_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_eq_in_process!(
        erlang::element(local_pid_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::element(external_pid_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_tuple_without_small_integer_index_is_bad_argument() {
    let mut process: Process = Default::default();
    let element_term = 1.into_process(&mut process);
    let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);
    let index = 0usize;
    let invalid_index_term = Term::arity(index);

    assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
    assert_eq_in_process!(
        erlang::element(tuple_term, invalid_index_term),
        Err(bad_argument!()),
        process
    );

    let valid_index_term: Term = index.into_process(&mut process);

    assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
    assert_eq_in_process!(
        erlang::element(tuple_term, valid_index_term),
        Ok(element_term),
        process
    );
}

#[test]
fn with_tuple_without_index_in_range_is_bad_argument() {
    let mut process: Process = Default::default();
    let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::element(empty_tuple_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_tuple_with_index_in_range_is_element() {
    let mut process: Process = Default::default();
    let element_term = 1.into_process(&mut process);
    let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);

    assert_eq_in_process!(
        erlang::element(tuple_term, 0.into_process(&mut process)),
        Ok(element_term),
        process
    );
}

#[test]
fn with_heap_binary_is_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_eq_in_process!(
        erlang::element(heap_binary_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}

#[test]
fn with_subbinary_is_bad_argument() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_eq_in_process!(
        erlang::element(subbinary_term, 0.into_process(&mut process)),
        Err(bad_argument!()),
        process
    );
}
