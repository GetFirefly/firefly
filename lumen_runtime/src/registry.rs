/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use alloc::sync::{Arc, Weak};

use hashbrown::HashMap;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::{AsTerm, Atom, Pid, Term};
use liblumen_alloc::{HeapAlloc, ProcessControlBlock};

use crate::process;

pub fn atom_to_process(name: &Atom) -> Option<Arc<ProcessControlBlock>> {
    let readable_registry = RW_LOCK_REGISTERED_BY_NAME.read();

    readable_registry
        .get(name)
        .and_then(|registered| match registered {
            Registered::Process(weak_process) => weak_process.upgrade(),
        })
}

pub fn names(process_control_block: &ProcessControlBlock) -> exception::Result {
    let mut acc = Term::NIL;
    let mut heap = process_control_block.acquire_heap();

    for name in RW_LOCK_REGISTERED_BY_NAME.read().keys() {
        let name_term = unsafe { name.as_term() };

        acc = heap.cons(name_term, acc)?
    }

    Ok(acc)
}

pub fn pid_to_process(pid: Pid) -> Option<Arc<ProcessControlBlock>> {
    RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID
        .read()
        .get(&pid)
        .and_then(|weak_process| weak_process.clone().upgrade())
}

pub fn pid_to_self_or_process(
    pid: Pid,
    process_arc: &Arc<ProcessControlBlock>,
) -> Option<Arc<ProcessControlBlock>> {
    if process_arc.pid() == pid {
        Some(process_arc.clone())
    } else {
        pid_to_process(pid)
    }
}

pub fn put_atom_to_process(
    name: Atom,
    arc_process_control_block: Arc<ProcessControlBlock>,
) -> bool {
    let writable_registry = RW_LOCK_REGISTERED_BY_NAME.write();

    if !writable_registry.contains_key(&name) {
        if process::register_in(arc_process_control_block, writable_registry, name) {
            true
        } else {
            false
        }
    } else {
        false
    }
}

pub fn put_pid_to_process(arc_process_control_block: &Arc<ProcessControlBlock>) {
    if let Some(_) = RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID.write().insert(
        arc_process_control_block.pid(),
        Arc::downgrade(&arc_process_control_block),
    ) {
        panic!("Process already registered with pid");
    }
}

pub fn unregister(name: &Atom) -> bool {
    match RW_LOCK_REGISTERED_BY_NAME.write().remove(name) {
        Some(Registered::Process(weak_process_control_block)) => {
            match weak_process_control_block.upgrade() {
                Some(arc_process_control_block) => {
                    let mut writable_registerd_name =
                        arc_process_control_block.registered_name.write();
                    *writable_registerd_name = None;

                    true
                }
                None => false,
            }
        }
        None => false,
    }
}

#[cfg_attr(test, derive(Debug))]
pub enum Registered {
    Process(Weak<ProcessControlBlock>),
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
    static ref RW_LOCK_WEAK_PROCESS_CONTROL_BLOCK_BY_PID: RwLock<HashMap<Pid, Weak<ProcessControlBlock>>> = Default::default();
}
