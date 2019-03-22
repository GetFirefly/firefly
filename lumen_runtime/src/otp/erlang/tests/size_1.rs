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

    assert_bad_argument!(erlang::size_1(atom_term, &mut process), &mut process);
}

#[test]
fn with_local_reference_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::local_reference(&mut process);

    assert_bad_argument!(erlang::size_1(tuple, &mut process), &mut process);
}

#[test]
fn with_empty_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_bad_argument!(erlang::size_1(Term::EMPTY_LIST, &mut process), &mut process);
}

#[test]
fn with_list_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = list_term(&mut process);

    assert_bad_argument!(erlang::size_1(list_term, &mut process), &mut process);
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term = 0usize.into_process(&mut process);

    assert_bad_argument!(
        erlang::size_1(small_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(erlang::size_1(big_integer_term, &mut process), &mut process);
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(erlang::size_1(float_term, &mut process), &mut process);
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(erlang::size_1(local_pid_term, &mut process), &mut process);
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::size_1(external_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_without_elements_is_zero() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);
    let zero_term = 0usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::size_1(empty_tuple_term, &mut process),
        Ok(zero_term),
        process
    );
}

#[test]
fn with_tuple_with_elements_is_element_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element_vec: Vec<Term> = (0..=2usize).map(|i| i.into_process(&mut process)).collect();
    let element_slice: &[Term] = element_vec.as_slice();
    let tuple_term = Term::slice_to_tuple(element_slice, &mut process);
    let arity_term = 3usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::size_1(tuple_term, &mut process),
        Ok(arity_term),
        process
    );
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(erlang::size_1(map_term, &mut process), &mut process);
}

#[test]
fn with_heap_binary_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);
    let byte_count_term = 3usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::size_1(heap_binary_term, &mut process),
        Ok(byte_count_term),
        process
    );
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let byte_count_term = 2usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::size_1(subbinary_term, &mut process),
        Ok(byte_count_term),
        process
    );
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);
    let byte_count_term = 2usize.into_process(&mut process);

    assert_eq_in_process!(
        erlang::size_1(subbinary_term, &mut process),
        Ok(byte_count_term),
        process
    );
}
