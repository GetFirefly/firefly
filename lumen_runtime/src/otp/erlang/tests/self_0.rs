use super::*;

use std::sync::{Arc, RwLock};

use crate::environment::{self, Environment};

#[test]
fn returns_process_pid() {
    let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
    let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
    let mut process = process_rw_lock.write().unwrap();

    assert_eq_in_process!(erlang::self_0(&process), process.pid, process);
    assert_eq_in_process!(
        erlang::self_0(&process),
        Term::local_pid(0, 0, &mut process).unwrap(),
        process
    );
}
