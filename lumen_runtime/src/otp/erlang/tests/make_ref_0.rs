use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn returns_a_unique_reference() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    let first_reference = erlang::make_ref_0(&mut process);
    let second_reference = erlang::make_ref_0(&mut process);

    assert_eq!(first_reference, first_reference);
    assert_ne!(first_reference, second_reference);
    assert_eq!(second_reference, second_reference);
}
