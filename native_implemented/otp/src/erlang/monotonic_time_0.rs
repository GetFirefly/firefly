#[cfg(test)]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::{monotonic, Unit::Native};

#[native_implemented::function(erlang:monotonic_time/0)]
pub async fn result(process: &Process) -> Term {
    let big_int = monotonic::time_in_unit(Native);

    process.integer(big_int).unwrap()
}
