#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{atoms, Term};

#[native_implemented::function(maps:find/2)]
pub fn result(process: &Process, key: Term, map: Term) -> Result<Term, NonNull<ErlangException>> {
    let map = term_try_into_map_or_badmap!(process, map)?;

    let result = match map.get(key) {
        Some(term) => {
            let ok = atoms::Ok;

            process.tuple_term_from_term_slice(&[ok, term])
        }
        None => atoms::Error.into(),
    };

    Ok(result)
}
