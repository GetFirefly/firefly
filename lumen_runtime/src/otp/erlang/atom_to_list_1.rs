// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(atom_to_list/1)]
pub fn native(process: &Process, atom: Term) -> exception::Result<Term> {
    let atom_atom: Atom = atom
        .try_into()
        .with_context(|| format!("atom ({}) is not an atom", atom))?;
    let chars = atom_atom.name().chars();
    let list = process.list_from_chars(chars)?;

    Ok(list)
}
