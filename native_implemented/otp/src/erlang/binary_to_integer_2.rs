#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::string_to_integer::base_string_to_integer;
use crate::runtime::binary_to_string::binary_to_string;

#[native_implemented::function(binary_to_integer/2)]
pub fn result(process: &Process, binary: Term, base: Term) -> exception::Result<Term> {
    let string: String = binary_to_string(binary)?;

    base_string_to_integer(process, base, "binary", '"', &string).map_err(From::from)
}
