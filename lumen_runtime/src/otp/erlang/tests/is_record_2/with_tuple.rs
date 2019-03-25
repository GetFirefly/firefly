use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_record_tag() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_eq_in_process!(
        erlang::is_record_2(term, record_tag, &mut process),
        Ok(true.into_process(&mut process)),
        &mut process
    );

    let other_atom = Term::str_to_atom("other_atom", DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::is_record_2(term, other_atom, &mut process),
        Ok(false.into_process(&mut process)),
        &mut process
    );
}

#[test]
fn with_local_reference_record_tag_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::local_reference(&mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list_record_tag_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::EMPTY_LIST;
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
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

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
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

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
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

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
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

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_errors_badard() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::local_pid(0, 0, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
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

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_tuple(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_map(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::slice_to_binary(&[], &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
    let record_tag = Term::subbinary(original, 0, 1, 1, 0, &mut process);
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_bad_argument!(
        erlang::is_record_2(term, record_tag, &mut process),
        &mut process
    );
}
