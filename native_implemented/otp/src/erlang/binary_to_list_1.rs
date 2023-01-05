#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:binary_to_list/1)]
pub fn result(process: &Process, binary: Term) -> Result<Term, NonNull<ErlangException>> {
    let bytes = process
        .bytes_from_binary(binary)
        .with_context(|| format!("binary ({})", binary))?;
    let byte_terms = bytes.iter().map(|byte| (*byte).into());

    Ok(process.list_from_iter(byte_terms))
}
