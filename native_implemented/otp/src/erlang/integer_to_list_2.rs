#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::integer_to_string::base_integer_to_string;

#[native_implemented::function(erlang:integer_to_list/2)]
pub fn result(process: &Process, integer: Term, base: Term) -> exception::Result<Term> {
    let string = base_integer_to_string(base, integer)?;
    let charlist = process.charlist_from_str(&string)?;

    Ok(charlist)
}
