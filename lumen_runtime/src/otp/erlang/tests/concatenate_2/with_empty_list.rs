use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

// The behavior here is weird to @KronicDeth and @bitwalker, but consistent with BEAM.
// See https://bugs.erlang.org/browse/ERL-898.

#[test]
fn with_atom_returns_atom() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::str_to_atom("term", DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_local_reference_returns_local_reference() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::local_reference(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_improper_list_returns_improper_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::cons(
        2.into_process(&mut process),
        3.into_process(&mut process),
        &mut process,
    );

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_small_integer_returns_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_big_integer_returns_big_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = <BigInt as Num>::from_str_radix("576460752303423490", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_float_returns_float() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::EMPTY_LIST;

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_local_pid_returns_local_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::local_pid(1, 2, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_external_pid_returns_external_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::external_pid(4, 5, 6, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_tuple_returns_tuple() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_map_is_returns_map_is() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::slice_to_map(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_heap_binary_returns_heap_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;
    let term = Term::slice_to_binary(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}

#[test]
fn with_subbinary_returns_subbinary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let list = Term::EMPTY_LIST;
    let term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(term),
        &mut process
    );
}
