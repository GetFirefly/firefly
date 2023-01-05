#[cfg(test)]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::{monotonic, system, Unit};

#[native_implemented::function(erlang:time_offset/1)]
pub fn result(process: &Process, unit: Term) -> Result<Term, NonNull<ErlangException>> {
    let unit_unit: Unit = unit.try_into()?;
    let system_time = system::time_in_unit(unit_unit);
    let monotonic_time = monotonic::time_in_unit(unit_unit);
    let term = process.integer(system_time - monotonic_time).unwrap();

    Ok(term)
}
