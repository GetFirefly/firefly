// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry;

#[native_implemented_function(unregister/1)]
pub fn result(name: Term) -> exception::Result<Term> {
    let atom = term_try_into_atom!(name)?;

    if registry::unregister(&atom) {
        Ok(true.into())
    } else {
        Err(anyhow!("name ({}) was not registered", name).into())
    }
}
