use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::process::Process;

pub struct Environment {
    pub process_by_pid_tagged: HashMap<usize, Arc<RwLock<Process>>>,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            process_by_pid_tagged: HashMap::new(),
        }
    }
}

#[cfg(test)]
impl Default for Environment {
    fn default() -> Environment {
        Environment::new()
    }
}

#[cfg(test)]
pub fn process(environment_rw_lock: Arc<RwLock<Environment>>) -> Arc<RwLock<Process>> {
    let process = Process::new(Arc::clone(&environment_rw_lock));
    let pid = process.pid;
    let process_rw_lock = Arc::new(RwLock::new(process));

    if let Some(_) = environment_rw_lock
        .write()
        .unwrap()
        .process_by_pid_tagged
        .insert(pid.tagged, Arc::clone(&process_rw_lock))
    {
        panic!("Process already registered with pid");
    }

    process_rw_lock
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::otp::erlang;

    #[test]
    fn different_processes_in_same_environment_have_different_pids() {
        let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();

        let first_process_rw_lock = process(Arc::clone(&environment_rw_lock));
        let first_process = first_process_rw_lock.write().unwrap();

        let second_process_rw_lock = process(Arc::clone(&environment_rw_lock));
        let second_process = second_process_rw_lock.write().unwrap();

        assert_ne!(
            erlang::self_0(&first_process),
            erlang::self_0(&second_process)
        );
    }
}
