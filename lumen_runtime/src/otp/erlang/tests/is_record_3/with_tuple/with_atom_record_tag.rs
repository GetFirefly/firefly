use super::*;

use std::sync::{Arc, RwLock};

use num_traits::Num;

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_size_errors_badarg() {
    with_size_errors_badarg(|_| Term::str_to_atom("1", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_size_errors_badarg() {
    with_size_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_size_errors_badarg() {
    with_size_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_size_errors_badarg() {
    with_size_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_size() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);
    let size = 1.into_process(&mut process);

    assert_eq!(erlang::is_record_3(term, record_tag, size), Ok(true.into()));

    let other_size = 2.into_process(&mut process);

    assert_eq!(
        erlang::is_record_3(term, record_tag, other_size),
        Ok(false.into())
    );
}

#[test]
fn with_big_integer_size_errors_badarg() {
    with_size_errors_badarg(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_size_errors_badarg() {
    with_size_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_size_errors_badard() {
    with_size_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_size_errors_badarg() {
    with_size_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuples_size_errors_badarg() {
    with_size_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_size_errors_badarg() {
    with_size_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_size_errors_badarg() {
    with_size_errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_size_errors_badarg() {
    with_size_errors_badarg(|mut process| {
        let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
        Term::subbinary(original, 0, 1, 1, 0, &mut process)
    });
}

fn with_size_errors_badarg<F>(size: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();
        let term = Term::slice_to_tuple(&[record_tag], &mut process);

        erlang::is_record_3(term, record_tag, size(&mut process))
    });
}
