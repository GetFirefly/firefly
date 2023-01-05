#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::unique_integer::unique_integer;

#[native_implemented::function(erlang:unique_integer/0)]
pub fn result(process: &Process) -> Term {
    unique_integer(process, Default::default())
}
