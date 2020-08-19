#[cfg(test)]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::time::{monotonic, system, Unit};

#[native_implemented::function(erlang:time_offset/1)]
pub fn result(process: &Process, unit: Term) -> exception::Result<Term> {
    let unit_unit: Unit = unit.try_into()?;
    let system_time = system::time_in_unit(unit_unit);
    let monotonic_time = monotonic::time_in_unit(unit_unit);
    let term = process.integer(system_time - monotonic_time);

    Ok(term)
}
