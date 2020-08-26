#[cfg(test)]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::unique_integer::unique_integer;

#[native_implemented::function(erlang:unique_integer/0)]
pub fn result(process: &Process) -> Term {
    unique_integer(process, Default::default())
}
