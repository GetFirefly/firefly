use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::atom::Existence::DoNotCare;
use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::str_to_atom("key", DoNotCare, &mut process).unwrap();
    let value = Term::str_to_atom("value", DoNotCare, &mut process).unwrap();
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::str_to_atom("non_key", DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_local_reference_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::local_reference(&mut process);
    let value = Term::local_reference(&mut process);
    let map_with_key = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map_with_key, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let map_without_key = Term::slice_to_map(&[], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map_without_key, &mut process),
        Ok(false.into_process(&mut process)),
        process
    )
}

#[test]
fn with_empty_list_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::EMPTY_LIST;
    let value = Term::EMPTY_LIST;
    let map_with_key = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map_with_key, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let map_without_key = Term::slice_to_map(&[], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map_without_key, &mut process),
        Ok(false.into_process(&mut process)),
        process
    )
}

#[test]
fn with_list_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);
    let value = Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::cons(2.into_process(&mut process), Term::EMPTY_LIST, &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_small_key_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = 0.into_process(&mut process);
    let value = 1.into_process(&mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = 2.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_big_key_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);
    let value = <BigInt as Num>::from_str_radix("576460752303423490", 10)
        .unwrap()
        .into_process(&mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = <BigInt as Num>::from_str_radix("576460752303423491", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_float_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = 1.0.into_process(&mut process);
    let value = 2.0.into_process(&mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = 3.0.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_local_key_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::local_pid(0, 1, &mut process).unwrap();
    let value = Term::local_pid(2, 3, &mut process).unwrap();
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::local_pid(4, 5, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_external_key_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::external_pid(1, 2, 3, &mut process).unwrap();
    let value = Term::external_pid(4, 5, 6, &mut process).unwrap();
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::external_pid(7, 8, 9, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_tuple_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);
    let value = Term::slice_to_tuple(&[1.into_process(&mut process)], &mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::slice_to_tuple(&[2.into_process(&mut process)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_heap_key_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let key = Term::slice_to_binary(&[0], &mut process);
    let value = Term::slice_to_binary(&[1], &mut process);
    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let non_key = Term::slice_to_binary(&[2], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_subbinary_key() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    // <<1::1, 2>>
    let key_original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
    let key = Term::subbinary(key_original, 0, 1, 1, 0, &mut process);

    // <<3::3, 4>>
    let value_original = Term::slice_to_binary(&[96, 0b0000_0000], &mut process);
    let value = Term::subbinary(value_original, 0, 3, 1, 0, &mut process);

    let map = Term::slice_to_map(&[(key, value)], &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(key, map, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    // <<5::5, 6>>
    let non_key_original = Term::slice_to_binary(&[40, 0b00110_000], &mut process);
    let non_key = Term::subbinary(non_key_original, 0, 5, 1, 0, &mut process);

    assert_eq_in_process!(
        erlang::is_map_key_2(non_key, map, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}
