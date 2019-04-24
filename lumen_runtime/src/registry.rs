/// Maps registered names (`Atom`) to `LocalPid` or `Port`
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::process::Process;
use crate::term::Term;

#[cfg_attr(test, derive(Debug, PartialEq))]
pub enum Registered {
    Process(Arc<Process>),
}

lazy_static! {
    pub static ref RW_LOCK_REGISTERED_BY_NAME: RwLock<HashMap<Term, Registered>> =
        Default::default();
}
