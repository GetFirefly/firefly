// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::erlang::string_to_integer::decimal_string_to_integer;
use crate::runtime::binary_to_string::binary_to_string;

#[native_implemented_function(binary_to_integer/1)]
pub fn result(process: &Process, binary: Term) -> exception::Result<Term> {
    let string: String = binary_to_string(binary)?;

    decimal_string_to_integer(process, "binary", '"', &string).map_err(From::from)
}
