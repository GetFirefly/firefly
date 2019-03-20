use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_returns_error_with_atom_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::str_to_atom("reason", Existence::DoNotCare, &mut process).unwrap();

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_empty_list_returns_error_with_empty_list_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::EMPTY_LIST;

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_list_returns_error_with_list_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = list_term(&mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_small_integer_returns_error_with_small_integer_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = 0usize.into_process(&mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_big_integer_returns_error_with_big_integer_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_float_returns_error_with_float_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = 1.0.into_process(&mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_local_pid_returns_error_with_local_pid_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::local_pid(0, 0, &mut process).unwrap();

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_external_pid_returns_error_with_external_pid_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_tuple_returns_error_with_tuple_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::slice_to_tuple(&[], &mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_map_returns_error_with_map_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let reason = Term::slice_to_map(&[], &mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_heap_binary_returns_error_with_heap_binary_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // :erlang.term_to_binary(:atom)
    let reason = Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}

#[test]
fn with_subbinary_returns_error_with_subbinary_reason() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
    let original_term = Term::slice_to_binary(
        &[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000],
        &mut process,
    );
    let reason = Term::subbinary(original_term, 0, 1, 8, 0, &mut process);

    assert_error!(erlang::error(reason), reason, &mut process);
}
