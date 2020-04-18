/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use std::sync::{Arc, Weak};

use dashmap::DashMap;
use lazy_static::lazy_static;

use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::exception;
use liblumen_alloc::Process;

lazy_static! {
    static ref REGISTERED_BY_NAME: DashMap<Atom, Registered> = Default::default();
    // Strong references are owned by the scheduler run queues
    static ref WEAK_PROCESS_CONTROL_BLOCK_BY_PID: DashMap<Pid, Weak<Process>> = Default::default();
}

pub fn atom_to_process(name: &Atom) -> Option<Arc<Process>> {
    REGISTERED_BY_NAME
        .get(name)
        .and_then(|registered| match registered.value() {
            Registered::Process(weak_process) => weak_process.upgrade(),
        })
}

pub fn names(process: &Process) -> exception::Result<Term> {
    let mut acc = Term::NIL;
    let mut heap = process.acquire_heap();

    for entry in REGISTERED_BY_NAME.iter() {
        let name_term: Term = entry.key().encode()?;

        let ptr = heap.cons(name_term, acc)?.as_ptr();
        acc = ptr.into();
    }

    Ok(acc)
}

pub fn pid_to_process(pid: &Pid) -> Option<Arc<Process>> {
    WEAK_PROCESS_CONTROL_BLOCK_BY_PID
        .get(pid)
        .and_then(|weak_process| weak_process.clone().upgrade())
}

pub fn pid_to_self_or_process(pid: Pid, process_arc: &Arc<Process>) -> Option<Arc<Process>> {
    if process_arc.pid() == pid {
        Some(process_arc.clone())
    } else {
        pid_to_process(&pid)
    }
}

pub fn put_atom_to_process(name: Atom, arc_process: Arc<Process>) -> bool {
    if !REGISTERED_BY_NAME.contains_key(&name) {
        register_in(arc_process, name)
    } else {
        false
    }
}

pub fn register_in(arc_process: Arc<Process>, name: Atom) -> bool {
    let mut writable_registered_name = arc_process.registered_name.write();

    if let None = *writable_registered_name {
        REGISTERED_BY_NAME.insert(name, Registered::Process(Arc::downgrade(&arc_process)));
        *writable_registered_name = Some(name);
        true
    } else {
        false
    }
}

pub fn put_pid_to_process(arc_process: &Arc<Process>) {
    if let Some(_) =
        WEAK_PROCESS_CONTROL_BLOCK_BY_PID.insert(arc_process.pid(), Arc::downgrade(&arc_process))
    {
        panic!("Process already registered with pid");
    }
}

pub fn unregister(name: &Atom) -> bool {
    match REGISTERED_BY_NAME.remove(name) {
        Some((_, Registered::Process(weak_process))) => match weak_process.upgrade() {
            Some(arc_process) => {
                let mut writable_registerd_name = arc_process.registered_name.write();
                *writable_registerd_name = None;

                true
            }
            None => false,
        },
        None => false,
    }
}

#[cfg_attr(test, derive(Debug))]
pub enum Registered {
    Process(Weak<Process>),
}

impl PartialEq for Registered {
    fn eq(&self, other: &Registered) -> bool {
        match (self, other) {
            (Registered::Process(self_weak_process), Registered::Process(other_weak_process)) => {
                Weak::ptr_eq(&self_weak_process, &other_weak_process)
            }
        }
    }
}
