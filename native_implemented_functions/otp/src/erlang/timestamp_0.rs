#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use lumen_rt_full::system::time::ErlangTimestamp;
use lumen_rt_full::time::{system, Unit::Microsecond};

#[native_implemented_function(timestamp/0)]
pub fn native(process: &Process) -> exception::Result<Term> {
    let big_int = system::time(Microsecond);
    let erlang_timestamp = ErlangTimestamp::from_microseconds(&big_int);

    process
        .tuple_from_slice(&[
            process.integer(erlang_timestamp.megaseconds as usize)?,
            process.integer(erlang_timestamp.seconds as usize)?,
            process.integer(erlang_timestamp.microseconds as usize)?,
        ])
        .map_err(|error| error.into())
}
