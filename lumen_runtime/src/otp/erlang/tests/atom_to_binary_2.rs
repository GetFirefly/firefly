use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_without_encoding_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, DoNotCare).unwrap();

    assert_badarg!(erlang::atom_to_binary_2(
        atom_term,
        0.into_process(&mut process),
        &mut process
    ));
}

#[test]
fn with_atom_with_invalid_encoding_atom_errors_badarg() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, DoNotCare).unwrap();
    let invalid_encoding_atom_term = Term::str_to_atom("invalid_encoding", DoNotCare).unwrap();

    assert_badarg!(erlang::atom_to_binary_2(
        atom_term,
        invalid_encoding_atom_term,
        &mut process
    ));
}

#[test]
fn with_atom_with_encoding_atom_returns_name_in_binary() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_name = "😈";
    let atom_term = Term::str_to_atom(atom_name, DoNotCare).unwrap();
    let latin1_atom_term = Term::str_to_atom("latin1", DoNotCare).unwrap();
    let unicode_atom_term = Term::str_to_atom("unicode", DoNotCare).unwrap();
    let utf8_atom_term = Term::str_to_atom("utf8", DoNotCare).unwrap();

    assert_eq!(
        erlang::atom_to_binary_2(atom_term, latin1_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process))
    );
    assert_eq!(
        erlang::atom_to_binary_2(atom_term, unicode_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
    );
    assert_eq!(
        erlang::atom_to_binary_2(atom_term, utf8_atom_term, &mut process),
        Ok(Term::slice_to_binary(atom_name.as_bytes(), &mut process)),
    );
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
fn with_tuple_errors_badarg() {
    errors_badarg(|mut process| {
        Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process,
        )
    });
}

#[test]
fn with_map_errors_badarg() {
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
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn errors_badarg<F>(atom: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let encoding_term = Term::str_to_atom("unicode", DoNotCare).unwrap();
        erlang::atom_to_binary_2(atom(&mut process), encoding_term, &mut process)
    });
}
