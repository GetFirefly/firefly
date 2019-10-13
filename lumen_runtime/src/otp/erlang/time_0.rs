#[cfg(test)]
mod test;

use crate::time::datetime;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(time/0)]
pub fn native(process: &Process) -> exception::Result {
    let time: [usize; 3] = datetime::local_time();

    process
        .tuple_from_slice(&[
            process.integer(time[0])?,
            process.integer(time[1])?,
            process.integer(time[2])?,
        ])
        .map_err(|error| error.into())
}
