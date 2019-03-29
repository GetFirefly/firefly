use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn returns_process_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let process = process_rw_lock.write().unwrap();

    assert_eq!(erlang::self_0(&process), process.pid);
    assert_eq!(erlang::self_0(&process), Term::local_pid(0, 0).unwrap());
}
