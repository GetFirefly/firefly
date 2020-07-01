#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(maps:find/2)]
pub fn result(process: &Process, key: Term, map: Term) -> exception::Result<Term> {
    let map = term_try_into_map_or_badmap!(process, map)?;

    let result = match map.get(key) {
        Some(term) => {
            let ok = atom!("ok");

            process.tuple_from_slice(&[ok, term])?
        }
        None => atom!("error"),
    };

    Ok(result.into())
}
