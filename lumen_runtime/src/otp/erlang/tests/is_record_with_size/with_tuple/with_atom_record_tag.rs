use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::atom::Existence::DoNotCare;
use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_size_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::str_to_atom("1", DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_empty_list_size_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::EMPTY_LIST;

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_list_size_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = list_term(&mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_small_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        Ok(true.into_process(&mut process)),
        process
    );

    let other_size = 2.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_record_with_size(term, record_tag, other_size, &mut process),
        Ok(false.into_process(&mut process)),
        process
    );
}

#[test]
fn with_big_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_float_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_errors_badard() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare, &mut process).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
    let size = Term::subbinary(original, 0, 1, 1, 0, &mut process);

    assert_bad_argument!(
        erlang::is_record_with_size(term, record_tag, size, &mut process),
        &mut process
    );
}
