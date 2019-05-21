use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

#[cfg(test)]
use crate::atom::Existence::DoNotCare;
#[cfg(test)]
use crate::code;
use crate::process::Process;
#[cfg(test)]
use crate::scheduler::Scheduler;
use crate::term::Term;

#[cfg(test)]
pub fn test_init() -> Arc<Process> {
    // During test allow multiple unregistered init processes because in tests, the `Scheduler`s
    // keep getting `Drop`ed as threads end.
    Scheduler::current().spawn_init()
}

#[cfg(test)]
pub fn test(parent_process: &Process) -> Arc<Process> {
    Scheduler::spawn(
        parent_process,
        Term::str_to_atom("erlang", DoNotCare).unwrap(),
        Term::str_to_atom("exit", DoNotCare).unwrap(),
        Term::slice_to_list(
            &[Term::str_to_atom("normal", DoNotCare).unwrap()],
            parent_process,
        ),
        code::apply_fn(),
    )
}

pub fn pid_to_process(pid: Term) -> Option<Arc<Process>> {
    RW_LOCK_WEAK_PROCESS_BY_PID
        .read()
        .unwrap()
        .get(&pid)
        .and_then(|weak_process| weak_process.clone().upgrade())
}

pub fn put_pid_to_process(process: &Arc<Process>) {
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
    // Strong references are owned by the scheduler run queues
    static ref RW_LOCK_WEAK_PROCESS_BY_PID: RwLock<HashMap<Term, Weak<Process>>> = Default::default();
}
