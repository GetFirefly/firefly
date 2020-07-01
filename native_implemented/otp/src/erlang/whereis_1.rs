#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::registry;

#[native_implemented::function(erlang:whereis/1)]
pub fn result(name: Term) -> exception::Result<Term> {
    let atom = term_try_into_atom!(name)?;
    let option = registry::atom_to_process(&atom).map(|arc_process| arc_process.pid());

    let term = match option {
        Some(pid) => pid.encode()?,
        None => atom!("undefined"),
    };

    Ok(term)
}
