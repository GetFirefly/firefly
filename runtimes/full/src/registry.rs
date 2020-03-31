/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use alloc::sync::{Arc, Weak};

use hashbrown::HashMap;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::process::alloc::TermAlloc;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::exception;
use liblumen_alloc::Process;

use crate::process;

pub fn atom_to_process(name: &Atom) -> Option<Arc<Process>> {
    let readable_registry = RW_LOCK_REGISTERED_BY_NAME.read();

    readable_registry
        .get(name)
        .and_then(|registered| match registered {
            Registered::Process(weak_process) => weak_process.upgrade(),
        })
}

pub fn names(process: &Process) -> exception::Result<Term> {
    let mut acc = Term::NIL;
    let mut heap = process.acquire_heap();

    for name in RW_LOCK_REGISTERED_BY_NAME.read().keys() {
        let name_term: Term = name.encode()?;

        let ptr = heap.cons(name_term, acc)?.as_ptr();
        acc = ptr.into();
    }

    Ok(acc)
}

pub fn pid_to_process(pid: &Pid) -> Option<Arc<Process>> {
    RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID
        .read()
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
    let writable_registry = RW_LOCK_REGISTERED_BY_NAME.write();

    if !writable_registry.contains_key(&name) {
        if process::register_in(arc_process, writable_registry, name) {
            true
        } else {
            false
        }
    } else {
        false
    }
}

pub fn put_pid_to_process(arc_process: &Arc<Process>) {
    if let Some(_) = RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID
        .write()
        .insert(arc_process.pid(), Arc::downgrade(&arc_process))
    {
        panic!("Process already registered with pid");
    }
}

pub fn unregister(name: &Atom) -> bool {
    match RW_LOCK_REGISTERED_BY_NAME.write().remove(name) {
        Some(Registered::Process(weak_process)) => match weak_process.upgrade() {
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

lazy_static! {
    static ref RW_LOCK_REGISTERED_BY_NAME: RwLock<HashMap<Atom, Registered>> = Default::default();
    // Strong references are owned by the scheduler run queues
    static ref RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID: RwLock<HashMap<Pid, Weak<Process>>> = Default::default();
}
