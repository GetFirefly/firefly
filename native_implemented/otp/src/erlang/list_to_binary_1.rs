#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::iolist_or_binary;

#[native_implemented::function(erlang:list_to_binary/1)]
pub fn result(process: &Process, iolist: Term) -> Result<Term, NonNull<ErlangException>> {
    match iolist {
        Term::Nil | Term::Cons(_) => {
            iolist_or_binary::to_binary(process, "iolist", iolist)
        }
        _ => Err(TypeError)
            .context(format!("iolist ({}) is not a list", iolist))
            .map_err(From::from),
    }
}
