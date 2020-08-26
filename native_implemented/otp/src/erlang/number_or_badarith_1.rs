#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:+/1)]
pub fn result(number: Term) -> exception::Result<Term> {
    if number.is_number() {
        Ok(number)
    } else {
        Err(badarith(Trace::capture()).into())
    }
}
