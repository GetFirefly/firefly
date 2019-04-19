use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::process::Process;
use crate::term::Term;

lazy_static! {
    static ref RW_LOCK_ARC_PROCESS_BY_PID: RwLock<HashMap<Term, Arc<Process>>> = Default::default();
}

#[cfg(test)]
pub fn new() -> Arc<Process> {
    let process = Process::new();
    let pid = process.pid;
    let process_arc = Arc::new(process);

    if let Some(_) = RW_LOCK_ARC_PROCESS_BY_PID
        .write()
        .unwrap()
        .insert(pid, process_arc.clone())
    {
        panic!("Process already registered with pid");
    }

    process_arc
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::otp::erlang;

    #[test]
    fn different_processes_in_same_environment_have_different_pids() {
        let first_process = new();
        let second_process = new();

        assert_ne!(
            erlang::self_0(&first_process),
            erlang::self_0(&second_process)
        );
    }
}
