#[cfg(all(not(feature = "runtime_minimal"), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(group_leader/0)]
pub fn result(process: &Process) -> Term {
    process.get_group_leader_pid_term()
}
