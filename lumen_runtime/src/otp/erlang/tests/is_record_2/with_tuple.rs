use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn with_atom_record_tag() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();
    let term = Term::slice_to_tuple(&[record_tag], &mut process);

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(true.into()));

    let other_atom = Term::str_to_atom("other_atom", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, other_atom), Ok(false.into()));
}

#[test]
fn with_local_reference_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_float_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_record_errors_badarg() {
    with_record_tag_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_record_tag_errors_badarg() {
    with_record_tag_errors_badarg(|mut process| {
        let original = Term::slice_to_binary(&[129, 0b0000_0000], &mut process);
        Term::subbinary(original, 0, 1, 1, 0, &mut process)
    });
}

fn with_record_tag_errors_badarg<F>(record_tag: F)
where
    F: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let record_tag = record_tag(&mut process);
        let term = Term::slice_to_tuple(&[record_tag], &mut process);

        erlang::is_record_2(term, record_tag)
    });
}
