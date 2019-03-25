use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

mod with_map;

#[test]
fn with_atom_errors_bad_map() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::str_to_atom("key", DoNotCare, &mut process).unwrap();
    let map = Term::str_to_atom("map", DoNotCare, &mut process).unwrap();

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_local_reference_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::local_reference(&mut process);
    let map = Term::local_reference(&mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_empty_list_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::EMPTY_LIST;
    let map = Term::EMPTY_LIST;

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_list_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process);
    let map = Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_small_integer_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = 1.into_process(&mut process);
    let map = 0.into_process(&mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_big_integer_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = <BigInt as Num>::from_str_radix("576460752303423490", 10)
        .unwrap()
        .into_process(&mut process);
    let map = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_float_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = 2.0.into_process(&mut process);
    let map = 1.0.into_process(&mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_local_pid_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::local_pid(2, 3, &mut process).unwrap();
    let map = Term::local_pid(0, 1, &mut process).unwrap();

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_external_pid_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::external_pid(4, 5, 6, &mut process).unwrap();
    let map = Term::external_pid(1, 2, 3, &mut process).unwrap();

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_tuple_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::slice_to_tuple(&[1.into_process(&mut process)], &mut process);
    let map = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let key = Term::slice_to_binary(&[1], &mut process);
    let map = Term::slice_to_binary(&[0], &mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}

#[test]
fn with_subbinary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    // <<3::3, 4>>
    let key_original = Term::slice_to_binary(&[96, 0b0100_0000], &mut process);
    let key = Term::subbinary(key_original, 0, 1, 1, 0, &mut process);

    // <<1::1, 2>>
    let map_original = Term::slice_to_binary(&[96, 0b0100_0000], &mut process);
    let map = Term::subbinary(map_original, 0, 1, 1, 0, &mut process);

    assert_bad_map!(
        erlang::is_map_key_2(key, map, &mut process),
        map,
        &mut process
    );
}
