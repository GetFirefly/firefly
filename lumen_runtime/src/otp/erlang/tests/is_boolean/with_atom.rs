use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};
use crate::process::IntoProcess;

#[test]
fn without_true_or_false_is_false() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_eq_in_process!(
        erlang::is_boolean(atom_term, &mut process),
        false.into_process(&mut process),
        process
    );
}

#[test]
fn with_true_is_true() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let term = true.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_boolean(term, &mut process),
        true.into_process(&mut process),
        process
    );
}

#[test]
fn with_false_is_true() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();
    let term = false.into_process(&mut process);

    assert_eq_in_process!(
        erlang::is_boolean(term, &mut process),
        true.into_process(&mut process),
        process
    );
}
