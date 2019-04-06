use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::process::{self, Process};
use crate::term::Term;

pub struct Environment {
    pid_counter: process::identifier::LocalCounter,
    pub process_by_pid_tagged: HashMap<usize, Arc<RwLock<Process>>>,
}

impl Environment {
    pub fn new() -> Environment {
        Environment {
            pid_counter: Default::default(),
            process_by_pid_tagged: HashMap::new(),
        }
    }

    pub fn next_pid(&mut self) -> Term {
        self.pid_counter.next().into()
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
