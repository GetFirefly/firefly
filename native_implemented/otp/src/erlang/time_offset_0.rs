#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::{monotonic, system, Unit::Native};

#[native_implemented::function(erlang:time_offset/0)]
pub fn result(process: &Process) -> Term {
    let system_time = system::time_in_unit(Native);
    let monotonic_time = monotonic::time_in_unit(Native);

    process.integer(system_time - monotonic_time).unwrap()
}
