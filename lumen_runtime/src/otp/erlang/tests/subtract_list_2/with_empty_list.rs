use super::*;

use num_traits::Num;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("term", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_returns_empty_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::EMPTY_LIST;
    let subtrahend = Term::EMPTY_LIST;

    assert_eq!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        Ok(Term::EMPTY_LIST)
    );
}

#[test]
fn with_list_returns_empty_list() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let minuend = Term::EMPTY_LIST;
    let subtrahend = Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);

    assert_eq!(
        erlang::subtract_list_2(minuend, subtrahend, &mut process),
        Ok(Term::EMPTY_LIST)
    );
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|mut process| {
        Term::cons(
            2.into_process(&mut process),
            3.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|mut process| 1.into_process(&mut process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423490", 10)
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
    errors_badarg(|_| Term::local_pid(1, 2).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|mut process| Term::external_pid(4, 5, 6, &mut process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 0, &mut process)
    });
}

fn errors_badarg<F>(subtrahend: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let minuend = Term::EMPTY_LIST;
        let subtrahend = subtrahend(&mut process);

        (minuend, subtrahend)
    });
}
