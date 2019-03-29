use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 0usize.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| {
        Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process,
        )
    });
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_without_encoding_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_badarg!(erlang::binary_to_existing_atom_2(
        heap_binary_term,
        0.into_process(&mut process)
    ));
}

#[test]
fn with_heap_binary_with_invalid_encoding_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);
    let invalid_encoding_term = Term::str_to_atom("invalid_encoding", DoNotCare).unwrap();

    assert_badarg!(erlang::binary_to_existing_atom_2(
        heap_binary_term,
        invalid_encoding_term
    ));
}

#[test]
fn with_heap_binary_with_valid_encoding_without_existing_atom_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("😈1".as_bytes(), &mut process);
    let latin1_atom_term = Term::str_to_atom("latin1", DoNotCare).unwrap();
    let unicode_atom_term = Term::str_to_atom("unicode", DoNotCare).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", DoNotCare).unwrap();

    assert_badarg!(erlang::binary_to_existing_atom_2(
        heap_binary_term,
        latin1_atom_term
    ));
    assert_badarg!(erlang::binary_to_existing_atom_2(
        heap_binary_term,
        unicode_atom_term
    ));
    assert_badarg!(erlang::binary_to_existing_atom_2(
        heap_binary_term,
        utf8_atom_term
    ));
}

#[test]
fn with_heap_binary_with_valid_encoding_with_existing_atom_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary("😈2".as_bytes(), &mut process);
    let latin1_atom_term = Term::str_to_atom("latin1", DoNotCare).unwrap();
    let unicode_atom_term = Term::str_to_atom("unicode", DoNotCare).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", DoNotCare).unwrap();
    let atom_term = Term::str_to_atom("😈2", DoNotCare).unwrap();

    assert_eq!(
        erlang::binary_to_existing_atom_2(heap_binary_term, latin1_atom_term),
        Ok(atom_term)
    );
    assert_eq!(
        erlang::binary_to_existing_atom_2(heap_binary_term, unicode_atom_term),
        Ok(atom_term)
    );
    assert_eq!(
        erlang::binary_to_existing_atom_2(heap_binary_term, utf8_atom_term),
        Ok(atom_term)
    );
}

#[test]
fn with_subbinary_with_bit_count_errors_badarg() {
    errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

#[test]
fn with_subbinary_without_bit_count_without_existing_atom_errors_badarg() {
    errors_badarg(|mut process| {
        let original = Term::slice_to_binary("😈🤘1".as_bytes(), &mut process);
        Term::subbinary(original, 4, 0, 5, 0, &mut process)
    });
}

#[test]
fn with_subbinary_without_bit_count_with_existing_atom_returns_atom_with_bytes() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary("😈🤘2".as_bytes(), &mut process);
    let binary = Term::subbinary(original, 4, 0, 5, 0, &mut process);
    let encoding = Term::str_to_atom("unicode", DoNotCare).unwrap();
    let atom_term = Term::str_to_atom("🤘2", DoNotCare).unwrap();

    assert_eq!(
        erlang::binary_to_existing_atom_2(binary, encoding),
        Ok(atom_term)
    )
}

fn errors_badarg<F>(binary: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|process| {
        let encoding = Term::str_to_atom("unicode", DoNotCare).unwrap();

        erlang::binary_to_existing_atom_2(binary(process), encoding)
    });
}
