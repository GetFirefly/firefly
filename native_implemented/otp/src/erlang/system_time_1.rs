#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::time::{system, Unit};

#[native_implemented::function(erlang:system_time/1)]
pub fn result(process: &Process, unit: Term) -> Result<Term, NonNull<ErlangException>> {
    let unit_unit: Unit = unit.try_into()?;
    let big_int = system::time_in_unit(unit_unit);
    let term = process.integer(big_int).unwrap();

    Ok(term)
}
