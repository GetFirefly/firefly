#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:self/0)]
pub fn result(process: &Process) -> Term {
    process.pid_term().unwrap()
}
