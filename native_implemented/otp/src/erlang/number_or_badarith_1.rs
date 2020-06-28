#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(+/1)]
pub fn result(number: Term) -> exception::Result<Term> {
    if number.is_number() {
        Ok(number)
    } else {
        Err(badarith(anyhow!("number ({}) is not an integer or a float", number).into()).into())
    }
}
