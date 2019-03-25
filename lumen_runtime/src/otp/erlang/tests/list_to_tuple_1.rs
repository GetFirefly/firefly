use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::str_to_atom("atom", DoNotCare, &mut process).unwrap();

    assert_bad_argument!(erlang::list_to_tuple_1(list, &mut process), &mut process);
}

#[test]
fn with_local_reference_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::local_reference(&mut process);

    assert_bad_argument!(erlang::list_to_tuple_1(list, &mut process), &mut process);
}

#[test]
fn with_empty_list_returns_empty_tuple() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::EMPTY_LIST;

    assert_eq_in_process!(
        erlang::list_to_tuple_1(list, &mut process),
        Ok(Term::slice_to_tuple(&[], &mut process)),
        &mut process
    );
}

#[test]
fn with_list_returns_tuple() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let first_element = 1.into_process(&mut process);
    let second_element = 2.into_process(&mut process);
    let third_element = 3.into_process(&mut process);
    let list = Term::cons(
        first_element,
        Term::cons(
            second_element,
            Term::cons(third_element, Term::EMPTY_LIST, &mut process),
            &mut process,
        ),
        &mut process,
    );

    assert_eq_in_process!(
        erlang::list_to_tuple_1(list, &mut process),
        Ok(Term::slice_to_tuple(
            &[first_element, second_element, third_element],
            &mut process
        )),
        &mut process
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list = Term::cons(
        0.into_process(&mut process),
        1.into_process(&mut process),
        &mut process,
    );

    assert_bad_argument!(erlang::list_to_tuple_1(list, &mut process), &mut process);
}

#[test]
fn with_nested_list_returns_tuple_with_list_element() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    // erlang doc: `[share, ['Ericsson_B', 163]]`
    let first_element = Term::str_to_atom("share", DoNotCare, &mut process).unwrap();
    let second_element = Term::cons(
        Term::str_to_atom("Ericsson_B", DoNotCare, &mut process).unwrap(),
        Term::cons(
            163.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process,
        ),
        &mut process,
    );
    let list = Term::cons(
        first_element,
        Term::cons(second_element, Term::EMPTY_LIST, &mut process),
        &mut process,
    );

    assert_eq_in_process!(
        erlang::list_to_tuple_1(list, &mut process),
        Ok(Term::slice_to_tuple(
            &[first_element, second_element],
            &mut process
        )),
        process
    )
}

#[test]
fn with_small_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let small_integer_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(small_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(big_integer_term, &mut process),
        &mut process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(float_term, &mut process),
        &mut process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let local_pid_term = Term::local_pid(0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_tuple_1(local_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_tuple_1(external_pid_term, &mut process),
        &mut process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(tuple_term, &mut process),
        &mut process
    );
}

#[test]
fn with_map_is_bad_argument() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let map_term = Term::slice_to_map(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(map_term, &mut process),
        &mut process
    );
}

#[test]
fn with_heap_binary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(heap_binary_term, &mut process),
        &mut process
    );
}

#[test]
fn with_subbinary_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_bad_argument!(
        erlang::list_to_tuple_1(subbinary_term, &mut process),
        &mut process
    );
}
