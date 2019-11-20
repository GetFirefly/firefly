// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(atom_to_list/1)]
pub fn native(process: &Process, atom: Term) -> exception::Result<Term> {
    match atom.decode().unwrap() {
        TypedTerm::Atom(atom) => {
            let chars = atom.name().chars();

            process.list_from_chars(chars).map_err(|error| error.into())
        }
        _ => Err(badarg!(process).into()),
    }
}
