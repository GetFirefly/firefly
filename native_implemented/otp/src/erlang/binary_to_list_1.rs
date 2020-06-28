#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

#[native_implemented::function(binary_to_list/1)]
pub fn result(process: &Process, binary: Term) -> exception::Result<Term> {
    let bytes = process
        .bytes_from_binary(binary)
        .with_context(|| format!("binary ({})", binary))?;
    let byte_terms = bytes.iter().map(|byte| (*byte).into());

    process.list_from_iter(byte_terms).map_err(From::from)
}
