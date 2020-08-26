#[cfg(test)]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::time::{monotonic, Unit::Native};

#[native_implemented::function(erlang:monotonic_time/0)]
pub fn result(process: &Process) -> Term {
    let big_int = monotonic::time_in_unit(Native);

    process.integer(big_int)
}
