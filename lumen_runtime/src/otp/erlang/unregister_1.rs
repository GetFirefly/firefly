// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::registry;

#[native_implemented_function(unregister/1)]
pub fn native(name: Term) -> exception::Result<Term> {
    let atom: Atom = name.try_into()?;

    if registry::unregister(&atom) {
        Ok(true.into())
    } else {
        Err(badarg!().into())
    }
}
