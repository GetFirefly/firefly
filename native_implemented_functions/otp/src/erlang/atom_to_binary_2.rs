// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::string::Encoding;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(atom_to_binary/2)]
pub fn result(process: &Process, atom: Term, encoding: Term) -> exception::Result<Term> {
    let atom_atom = term_try_into_atom!(atom)?;
    let _: Encoding = encoding.try_into()?;
    let binary = process.binary_from_str(atom_atom.name())?;

    Ok(binary)
}
