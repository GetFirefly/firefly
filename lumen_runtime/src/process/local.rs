use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

use crate::process::Process;
use crate::term::Term;

#[cfg(test)]
pub fn new() -> Arc<Process> {
    crate::scheduler::Scheduler::current().new_process()
}

pub fn pid_to_process(pid: Term) -> Option<Arc<Process>> {
    RW_LOCK_WEAK_PROCESS_BY_PID
        .read()
        .unwrap()
        .get(&pid)
        .and_then(|weak_process| weak_process.clone().upgrade())
}

#[cfg(test)]
pub fn put_pid_to_process(process: Arc<Process>) {
    if let Some(_) = RW_LOCK_WEAK_PROCESS_BY_PID
        .write()
        .unwrap()
        .insert(process.pid, Arc::downgrade(&process))
    {
        panic!("Process already registerd with pid");
    }
}

pub fn pid_to_self_or_process(pid: Term, process_arc: &Arc<Process>) -> Option<Arc<Process>> {
    if process_arc.pid.tagged == pid.tagged {
        Some(process_arc.clone())
    } else {
        pid_to_process(pid)
    }
}

lazy_static! {
    // Strong references are owned by the schedulers
    static ref RW_LOCK_WEAK_PROCESS_BY_PID: RwLock<HashMap<Term, Weak<Process>>> = Default::default();
}
