#[cfg(test)]
mod test;

use crate::time::datetime;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(universaltime/0)]
pub fn native(process: &Process) -> exception::Result {
    let now: [usize; 6] = datetime::utc_now();

    let date_tuple = process.tuple_from_slice(&[
        process.integer(now[0])?,
        process.integer(now[1])?,
        process.integer(now[2])?,
    ])?;
    let time_tuple = process.tuple_from_slice(&[
        process.integer(now[3])?,
        process.integer(now[4])?,
        process.integer(now[5])?,
    ])?;

    return process
        .tuple_from_slice(&[date_tuple, time_tuple])
        .map_err(|error| error.into());
}
