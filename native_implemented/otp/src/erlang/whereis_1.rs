#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::term::{atoms, Term};

use crate::runtime::registry;

#[native_implemented::function(erlang:whereis/1)]
pub fn result(name: Term) -> Result<Term, NonNull<ErlangException>> {
    let atom = term_try_into_atom!(name)?;
    let option = registry::atom_to_process(&atom).map(|arc_process| arc_process.pid());

    let term = match option {
        Some(pid) => pid.encode()?,
        None => atoms::Undefined.into(),
    };

    Ok(term)
}
