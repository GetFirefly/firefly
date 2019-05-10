/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use std::collections::HashMap;
use std::sync::{RwLock, Weak};

use crate::process::Process;
use crate::term::Term;

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
