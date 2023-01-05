#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:group_leader/0)]
pub fn result(process: &Process) -> Term {
    process.get_group_leader_pid_term()
}
