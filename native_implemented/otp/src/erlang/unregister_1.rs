#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::registry;

#[native_implemented::function(unregister/1)]
pub fn result(name: Term) -> exception::Result<Term> {
    let atom = term_try_into_atom!(name)?;

    if registry::unregister(&atom) {
        Ok(true.into())
    } else {
        Err(anyhow!("name ({}) was not registered", name).into())
    }
}
