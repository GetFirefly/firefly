#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang::iolist_or_binary;

#[native_implemented::function(list_to_binary/1)]
pub fn result(process: &Process, iolist: Term) -> exception::Result<Term> {
    match iolist.decode()? {
        TypedTerm::Nil | TypedTerm::List(_) => {
            iolist_or_binary::to_binary(process, "iolist", iolist)
        }
        _ => Err(TypeError)
            .context(format!("iolist ({}) is not a list", iolist))
            .map_err(From::from),
    }
}
