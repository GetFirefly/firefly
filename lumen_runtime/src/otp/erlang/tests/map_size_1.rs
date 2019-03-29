use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::str_to_atom("atom", DoNotCare).unwrap();

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_local_reference_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::local_reference(&mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_empty_list_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::EMPTY_LIST;

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_list_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = list_term(&mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_small_integer_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = 0usize.into_process(&mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_big_integer_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_badarg!(erlang::tuple_size_1(big_integer_term, &mut process));
}

#[test]
fn with_float_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = 1.0.into_process(&mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_local_pid_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::local_pid(0, 0).unwrap();

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_external_pid_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_tuple_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::slice_to_tuple(&[], &mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_map_without_elements_is_zero() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::slice_to_map(&[], &mut process);

    assert_eq!(
        erlang::map_size_1(map, &mut process),
        Ok(0.into_process(&mut process))
    );
}

#[test]
fn with_map_with_elements_is_element_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::slice_to_map(
        &[
            (
                Term::str_to_atom("one", DoNotCare).unwrap(),
                1.into_process(&mut process),
            ),
            (
                Term::str_to_atom("two", DoNotCare).unwrap(),
                2.into_process(&mut process),
            ),
        ],
        &mut process,
    );

    assert_eq!(
        erlang::map_size_1(map, &mut process),
        Ok(2.into_process(&mut process))
    );
}
#[test]
fn with_heap_binary_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map = Term::slice_to_binary(&[0, 1, 2], &mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}

#[test]
fn with_subbinary_errors_badmap() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary(&[0b0111_1111, 0b1100_0000], &mut process);
    let map = Term::subbinary(original, 0, 1, 1, 1, &mut process);

    assert_badmap!(erlang::map_size_1(map, &mut process), map, &mut process);
}
