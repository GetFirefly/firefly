use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};

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
    errors_badarg(|mut process| 0.into_process(&mut process));
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
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_returns_list_of_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let bit_string = Term::slice_to_binary(&[0], &mut process);

    assert_eq!(
        erlang::bitstring_to_list_1(bit_string, &mut process),
        Ok(Term::cons(
            0.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process
        ))
    );
}

#[test]
fn with_subbinary_without_bit_count_returns_list_of_integer() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
    let bit_string = Term::subbinary(original, 1, 0, 1, 0, &mut process);

    assert_eq!(
        erlang::bitstring_to_list_1(bit_string, &mut process),
        Ok(Term::cons(
            1.into_process(&mut process),
            Term::EMPTY_LIST,
            &mut process
        ))
    );
}

#[test]
fn with_subbinary_with_bit_count_returns_list_of_integer_with_bitstring_for_bit_count() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let original = Term::slice_to_binary(&[0, 1, 0b010], &mut process);
    let bit_string = Term::subbinary(original, 0, 0, 2, 3, &mut process);

    assert_eq!(
        erlang::bitstring_to_list_1(bit_string, &mut process),
        Ok(Term::cons(
            0.into_process(&mut process),
            Term::cons(
                1.into_process(&mut process),
                Term::subbinary(
                    Term::slice_to_binary(&[0, 1, 2], &mut process),
                    2,
                    0,
                    0,
                    3,
                    &mut process
                ),
                &mut process
            ),
            &mut process
        ))
    );
}

fn errors_badarg<F>(bit_string: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        erlang::bitstring_to_list_1(bit_string(&mut process), &mut process)
    });
}
