#[cfg(test)]
mod test;

use crate::runtime::time::datetime;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:time/0)]
pub fn result(process: &Process) -> Term {
    let time: [usize; 3] = datetime::local_time();

    process.tuple_from_slice(&[
        process.integer(time[0]),
        process.integer(time[1]),
        process.integer(time[2]),
    ])
}
