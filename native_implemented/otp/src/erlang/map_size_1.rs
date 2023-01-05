#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:map_size/1)]
pub fn result(process: &Process, map: Term) -> Result<Term, NonNull<ErlangException>> {
    let boxed_map = term_try_into_map_or_badmap!(process, map)?;
    let len = boxed_map.len();
    let len_term = process.integer(len).unwrap();

    Ok(len_term)
}
