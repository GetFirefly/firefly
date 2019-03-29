use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};

#[test]
fn with_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::str_to_atom("atom", DoNotCare).unwrap();

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_local_reference_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::local_reference(&mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_empty_list_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_badarg!(erlang::insert_element_3(
        Term::EMPTY_LIST,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_list_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = list_term(&mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_small_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = 0.into_process(&mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_big_integer_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_float_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = 1.0.into_process(&mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_local_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::local_pid(0, 0).unwrap();

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_external_pid_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_badarg!(erlang::insert_element_3(
        tuple,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_tuple_without_small_integer_index_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::slice_to_tuple(
        &[1.into_process(&mut process), 3.into_process(&mut process)],
        &mut process,
    );
    let index = 2usize;
    let invalid_index_term = Term::arity(index);

    assert_ne!(invalid_index_term.tag(), SmallInteger);
    assert_badarg!(erlang::insert_element_3(
        tuple,
        invalid_index_term,
        0.into_process(&mut process),
        &mut process
    ));

    let valid_index_term: Term = index.into_process(&mut process);

    assert_eq!(valid_index_term.tag(), SmallInteger);
    assert_eq!(
        erlang::insert_element_3(
            tuple,
            valid_index_term,
            2.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[
                1.into_process(&mut process),
                2.into_process(&mut process),
                3.into_process(&mut process)
            ],
            &mut process
        ))
    );
}

#[test]
fn with_tuple_without_index_in_range_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::slice_to_tuple(&[], &mut process);

    assert_badarg!(erlang::insert_element_3(
        tuple,
        2.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_tuple_with_index_in_range_returns_tuple_with_new_element_at_index() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::slice_to_tuple(
        &[1.into_process(&mut process), 3.into_process(&mut process)],
        &mut process,
    );

    assert_eq!(
        erlang::insert_element_3(
            tuple,
            2.into_process(&mut process),
            2.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[
                1.into_process(&mut process),
                2.into_process(&mut process),
                3.into_process(&mut process)
            ],
            &mut process
        ))
    );
}

#[test]
fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple = Term::slice_to_tuple(&[1.into_process(&mut process)], &mut process);

    assert_eq!(
        erlang::insert_element_3(
            tuple,
            2.into_process(&mut process),
            3.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[1.into_process(&mut process), 3.into_process(&mut process)],
            &mut process
        ))
    )
}

#[test]
fn with_map_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let term = Term::slice_to_map(&[], &mut process);

    assert_badarg!(erlang::insert_element_3(
        term,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_heap_binary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_badarg!(erlang::insert_element_3(
        heap_binary_term,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_subbinary_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_badarg!(erlang::insert_element_3(
        subbinary_term,
        0.into_process(&mut process),
        0.into_process(&mut process),
        &mut process
    ));
}
