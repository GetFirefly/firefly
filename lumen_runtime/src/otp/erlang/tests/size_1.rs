use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_errors_badarg() {
    errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 0usize.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_without_elements_is_zero() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);
    let zero_term = 0usize.into_process(&mut process);

    assert_eq!(
        erlang::size_1(empty_tuple_term, &mut process),
        Ok(zero_term)
    );
}

#[test]
fn with_tuple_with_elements_is_element_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let element_vec: Vec<Term> = (0..=2usize).map(|i| i.into_process(&mut process)).collect();
    let element_slice: &[Term] = element_vec.as_slice();
    let tuple_term = Term::slice_to_tuple(element_slice, &mut process);
    let arity_term = 3usize.into_process(&mut process);

    assert_eq!(erlang::size_1(tuple_term, &mut process), Ok(arity_term));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);
    let byte_count_term = 3usize.into_process(&mut process);

    assert_eq!(
        erlang::size_1(heap_binary_term, &mut process),
        Ok(byte_count_term)
    );
}

#[test]
fn with_subbinary_with_bit_count_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);
    let byte_count_term = 2usize.into_process(&mut process);

    assert_eq!(
        erlang::size_1(subbinary_term, &mut process),
        Ok(byte_count_term)
    );
}

#[test]
fn with_subbinary_without_bit_count_is_byte_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 0, &mut process);
    let byte_count_term = 2usize.into_process(&mut process);

    assert_eq!(
        erlang::size_1(subbinary_term, &mut process),
        Ok(byte_count_term)
    );
}

fn errors_badarg<F>(binary_or_tuple: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| erlang::size_1(binary_or_tuple(&mut process), &mut process));
}
