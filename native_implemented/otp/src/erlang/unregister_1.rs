#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

use crate::runtime::registry;

#[native_implemented::function(erlang:unregister/1)]
pub fn result(name: Term) -> Result<Term, NonNull<ErlangException>> {
    let atom = term_try_into_atom!(name)?;

    if registry::unregister(&atom) {
        Ok(true.into())
    } else {
        Err(anyhow!("name ({}) was not registered", name).into())
    }
}
