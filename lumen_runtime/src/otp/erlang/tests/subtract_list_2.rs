use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

mod with_empty_list;
mod with_list;

#[test]
fn with_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::str_to_atom("list", DoNotCare, &mut process).unwrap();
    let subtrahend = Term::str_to_atom("term", DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_local_reference_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::local_reference(&mut process);
    let subtrahend = Term::local_reference(&mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::cons(
        0.into_process(&mut process),
        1.into_process(&mut process),
        &mut process,
    );
    let subtrahend = Term::cons(
        2.into_process(&mut process),
        3.into_process(&mut process),
        &mut process,
    );

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = 0.into_process(&mut process);
    let subtrahend = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);
    let subtrahend = <BigInt as Num>::from_str_radix("576460752303423490", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_float_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = 1.0.into_process(&mut process);
    let subtrahend = 2.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::local_pid(0, 1, &mut process).unwrap();
    let subtrahend = Term::local_pid(1, 2, &mut process).unwrap();

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::external_pid(1, 2, 3, &mut process).unwrap();
    let subtrahend = Term::external_pid(4, 5, 6, &mut process).unwrap();

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::slice_to_tuple(&[], &mut process);
    let subtrahend = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::slice_to_map(&[], &mut process);
    let subtrahend = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::slice_to_binary(&[], &mut process);
    let subtrahend = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let minuend = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let subtrahend = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);

    assert_bad_argument!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        &mut process
    );
}
