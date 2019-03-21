use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};

#[test]
fn with_atom_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::bitstring_to_list_1(atom_term, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_bad_argument!(
        erlang::bitstring_to_list_1(Term::EMPTY_LIST, &mut process),
        &mut process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(list_term, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term: Term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(small_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(big_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(float_term, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::bitstring_to_list_1(local_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::bitstring_to_list_1(external_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(tuple_term, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::bitstring_to_list_1(map_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_returns_list_of_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);

    assert_eq_in_process!(
        erlang::bitstring_to_list_1(heap_binary_term, &mut process),
        Ok(Term::cons(
            0.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process
        )),
        process
    );
}

#[test]
fn with_subbinary_without_bit_count_returns_list_of_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 1, 0, 1, 0, &mut process);

    assert_eq_in_process!(
        erlang::bitstring_to_list_1(subbinary_term, &mut process),
        Ok(Term::cons(
            1.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process
        )),
        process
    );
}

#[test]
fn with_subbinary_with_bit_count_returns_list_of_integer_with_bitstring_for_bit_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 0, 2, 3, &mut process);

    assert_eq_in_process!(
        erlang::bitstring_to_list_1(subbinary_term, &mut process),
        Ok(Term::cons(
            0.into_process(&mut process),
            Term::cons(
                1.into_process(&mut process),
                Term::subbinary(
                    Term::slice_to_binary(&[0, 1, 2], &mut process),
                    2,
                    0,
                    0,
                    3,
                    &mut process
                ),
                &mut process
            ),
            &mut process
        )),
        process
    );
}
