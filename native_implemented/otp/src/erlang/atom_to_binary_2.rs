use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:atom_to_binary/2)]
pub fn result(
    process: &Process,
    atom: Term,
    encoding: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let atom_atom = term_try_into_atom!(atom)?;
    let _: Encoding = encoding.try_into()?;
    let binary = process.binary_from_str(atom_atom.as_str());

    Ok(binary)
}
