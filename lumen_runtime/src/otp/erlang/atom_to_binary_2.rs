// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::string::Encoding;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(atom_to_binary/2)]
pub fn native(process: &Process, atom: Term, encoding: Term) -> exception::Result {
    match atom.decode().unwrap() {
        TypedTerm::Atom(atom) => {
            let _: Encoding = encoding.try_into()?;
            let binary = process.binary_from_str(atom.name())?;

            Ok(binary)
        }
        _ => Err(badarg!().into()),
    }
}
