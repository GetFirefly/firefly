#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::charlist_to_string::charlist_to_string;
use crate::erlang::string_to_float::string_to_float;

#[native_implemented::function(erlang:list_to_float/1)]
pub fn result(process: &Process, list: Term) -> exception::Result<Term> {
    let string = charlist_to_string(list)?;

    string_to_float(process, "list", &string, '\'').map_err(From::from)
}
