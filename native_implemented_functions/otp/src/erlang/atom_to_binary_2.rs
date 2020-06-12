#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::string::Encoding;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(atom_to_binary/2)]
pub fn result(process: &Process, atom: Term, encoding: Term) -> exception::Result<Term> {
    let atom_atom = term_try_into_atom!(atom)?;
    let _: Encoding = encoding.try_into()?;
    let binary = process.binary_from_str(atom_atom.name())?;

    Ok(binary)
}
