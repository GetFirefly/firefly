#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::string_to_integer::decimal_string_to_integer;
use crate::runtime::binary_to_string::binary_to_string;

#[native_implemented::function(binary_to_integer/1)]
pub fn result(process: &Process, binary: Term) -> exception::Result<Term> {
    let string: String = binary_to_string(binary)?;

    decimal_string_to_integer(process, "binary", '"', &string).map_err(From::from)
}
