#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::backtrace::Trace;
use firefly_rt::process::Process;
use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

#[native_implemented::function(maps:get/2)]
pub fn result(process: &Process, key: Term, map: Term) -> Result<Term, NonNull<ErlangException>> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;

    match boxed_map.get(key) {
        Some(value) => Ok(value),
        None => Err(badkey(
            process,
            key,
            Trace::capture(),
            anyhow!("map ({}) does not have key ({})", map, key).into(),
        )),
    }
}
