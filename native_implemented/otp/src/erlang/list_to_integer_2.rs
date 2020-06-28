#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::list_to_string::list_to_string;
use crate::erlang::string_to_integer::base_string_to_integer;

#[native_implemented::function(list_to_integer/2)]
pub fn result(process: &Process, list: Term, base: Term) -> exception::Result<Term> {
    let string: String = list_to_string(list)?;

    base_string_to_integer(process, base, "list", '\'', &string).map_err(From::from)
}
