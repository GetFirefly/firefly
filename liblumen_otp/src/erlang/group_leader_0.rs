#[cfg(test)]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(group_leader/0)]
pub fn native(process: &Process) -> Term {
    process.get_group_leader_pid_term()
}
