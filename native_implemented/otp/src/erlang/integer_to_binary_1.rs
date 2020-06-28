#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::integer_to_string::decimal_integer_to_string;

#[native_implemented::function(integer_to_binary/1)]
pub fn result(process: &Process, integer: Term) -> exception::Result<Term> {
    let string = decimal_integer_to_string(integer)?;
    let binary = process.binary_from_str(&string)?;

    Ok(binary)
}
