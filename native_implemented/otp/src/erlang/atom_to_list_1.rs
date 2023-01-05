use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:atom_to_list/1)]
pub fn result(process: &Process, atom: Term) -> Result<Term, NonNull<ErlangException>> {
    let atom_atom = term_try_into_atom!(atom)?;
    let chars = atom_atom.as_str().chars();
    let list = process.list_from_chars(chars);

    Ok(list)
}
