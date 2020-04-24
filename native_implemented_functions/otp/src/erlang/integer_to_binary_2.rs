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

use crate::erlang::integer_to_string::base_integer_to_string;

#[native_implemented_function(integer_to_binary/2)]
pub fn result(process: &Process, integer: Term, base: Term) -> exception::Result<Term> {
    let string = base_integer_to_string(base, integer)?;
    let binary = process.binary_from_str(&string)?;

    Ok(binary)
}
