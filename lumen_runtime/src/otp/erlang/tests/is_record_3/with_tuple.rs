use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

mod with_atom_record_tag;

#[test]
fn with_empty_list_record_tag_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::EMPTY_LIST;
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_list_record_tag_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = list_term(&mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = 0.into_process(&mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_float_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = 1.0.into_process(&mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::local_pid(0, 0, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::external_pid(1, 0, 0, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_tuple(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_map_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_map(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_binary(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
    let record_tag = Term::subbinary(original, 0, 1, 1, 0, &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_3(term, record_tag, size, &mut process),
        &mut process
    );
}
