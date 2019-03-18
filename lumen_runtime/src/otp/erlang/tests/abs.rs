use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(erlang::abs(atom_term, &mut process), process);
}

#[test]
fn with_heap_binary_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0], &mut process);

    assert_bad_argument!(erlang::abs(heap_binary_term, &mut process), process);
}

#[test]
fn with_subbinary_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_bad_argument!(erlang::abs(subbinary_term, &mut process), process);
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_bad_argument!(erlang::abs(Term::EMPTY_LIST, &mut process), process);
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);

    assert_bad_argument!(erlang::abs(list_term, &mut process), process);
}

#[test]
fn with_small_integer_that_is_negative_returns_positive() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let negative: isize = -1;
    let negative_term = negative.into_process(&mut process);

    let positive = -negative;
    let positive_term = positive.into_process(&mut process);

    assert_eq_in_process!(
        erlang::abs(negative_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_small_integer_that_is_positive_returns_self() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let positive_term = 1usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::abs(positive_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_big_integer_that_is_negative_returns_positive() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let negative: isize = small::MIN - 1;
    let negative_term = negative.into_process(&mut process);

    assert_eq!(negative_term.tag(), Tag::Boxed);

    let unboxed_negative_term: &Term = negative_term.unbox_reference();

    assert_eq!(unboxed_negative_term.tag(), Tag::BigInteger);

    let positive = -negative;
    let positive_term = positive.into_process(&mut process);

    assert_eq_in_process!(
        erlang::abs(negative_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_big_integer_that_is_positive_return_self() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let positive_term: Term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq!(positive_term.tag(), Tag::Boxed);

    let unboxed_positive_term: &Term = positive_term.unbox_reference();

    assert_eq!(unboxed_positive_term.tag(), Tag::BigInteger);

    assert_eq_in_process!(
        erlang::abs(positive_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_float_that_is_negative_returns_positive() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let negative = -1.0;
    let negative_term = negative.into_process(&mut process);

    assert_eq!(negative_term.tag(), Tag::Boxed);

    let unboxed_negative_term: &Term = negative_term.unbox_reference();

    assert_eq!(unboxed_negative_term.tag(), Tag::Float);

    let positive = -negative;
    let positive_term = positive.into_process(&mut process);

    assert_eq_in_process!(
        erlang::abs(negative_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_float_that_is_positive_return_self() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let positive_term: Term = 1.0.into_process(&mut process);

    assert_eq!(positive_term.tag(), Tag::Boxed);

    let unboxed_positive_term: &Term = positive_term.unbox_reference();

    assert_eq!(unboxed_positive_term.tag(), Tag::Float);

    assert_eq_in_process!(
        erlang::abs(positive_term, &mut process),
        Ok(positive_term),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_bad_argument!(erlang::abs(local_pid_term, &mut process), process);
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(erlang::abs(external_pid_term, &mut process), process);
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(erlang::abs(tuple_term, &mut process), process);
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(erlang::abs(map_term, &mut process), process);
}
