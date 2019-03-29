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
fn with_empty_list_is_zero() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let zero_term = 0.into_process(&mut process);

    assert_eq!(
        erlang::length_1(Term::EMPTY_LIST, &mut process),
        Ok(zero_term)
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|mut process| {
        let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
        let tail_term = Term::str_to_atom("tail", DoNotCare).unwrap();
        Term::cons(head_term, tail_term, &mut process)
    });
}

#[test]
fn with_list_is_length_1() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let list_term = (0..=2).rfold(Term::EMPTY_LIST, |acc, i| {
        Term::cons(i.into_process(&mut process), acc, &mut process)
    });

    assert_eq!(
        erlang::length_1(list_term, &mut process),
        Ok(3.into_process(&mut process))
    );
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
fn with_heap_binary_is_false() {
    errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_is_false() {
    errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn errors_badarg<F>(list: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| erlang::length_1(list(&mut process), &mut process));
}
