use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_return_improper_list_with_atom_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::str_to_atom("term", DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_local_reference_returns_improper_list_with_local_reference_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::local_reference(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_improper_list_return_improper_list_with_improper_list_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::cons(
        1.into_process(&mut process),
        2.into_process(&mut process),
        &mut process,
    );

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_small_integer_returns_improper_list_with_small_integer_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = 1.into_process(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_big_integer_returns_improper_list_with_big_integer_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = <BigInt as Num>::from_str_radix("576460752303423490", 10)
        .unwrap()
        .into_process(&mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_float_returns_improper_list_with_float_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_local_pid_returns_improper_list_with_local_pid_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::local_pid(1, 2, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_external_pid_returns_improper_list_with_external_pid_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::external_pid(4, 5, 6, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_tuple_returns_improper_list_with_tuple_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::slice_to_tuple(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_map_returns_improper_list_with_map_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::slice_to_map(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_heap_binary_returns_improper_list_with_heap_binary_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::slice_to_binary(&[], &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}

#[test]
fn with_subbinary_returns_improper_list_with_subbinary_as_tail() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let element = 0.into_process(&mut process);
    let list = Term::cons(element, Term::EMPTY_LIST, &mut process);
    let term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);

    assert_eq_in_process!(
        erlang::concatenate_2(list, term, &mut process),
        Ok(Term::cons(element, term, &mut process)),
        &mut process
    );
}
