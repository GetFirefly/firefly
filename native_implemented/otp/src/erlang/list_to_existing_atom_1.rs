#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::{Atom, Term};

use crate::erlang::list_to_string::list_to_string;

#[native_implemented::function(erlang:list_to_existing_atom/1)]
pub fn result(string: Term) -> Result<Term, NonNull<ErlangException>> {
    let string_string = list_to_string(string)?;
    let atom = Atom::try_from_str_existing(string_string)
        .with_context(|| format!("string ({})", string))?;

    atom.encode().map_err(From::from)
}
