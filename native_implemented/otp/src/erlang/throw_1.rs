#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(throw/1)]
pub fn result(reason: Term) -> exception::Result<Term> {
    Err(throw(reason, None, anyhow!("explicit throw from Erlang").into()).into())
}
