/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};

use crate::atom::Existence::Exists;
use crate::process::Process;
use crate::term::Term;

pub fn name_to_process(name: &str) -> Option<Arc<Process>> {
    // If the atom doesn't exist it can't be registered, so don't need to use `DoNotCare`
    Term::str_to_atom(name, Exists).and_then(|name_term| {
        let readable_registry = RW_LOCK_REGISTERED_BY_NAME.read().unwrap();

        readable_registry
            .get(&name_term)
            .and_then(|registered| match registered {
                Registered::Process(weak_process) => weak_process.upgrade(),
            })
    })
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
    pub static ref RW_LOCK_REGISTERED_BY_NAME: RwLock<HashMap<Term, Registered>> =
        Default::default();
}
