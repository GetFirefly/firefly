// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::erlang::iolist_or_binary;

#[native_implemented_function(list_to_binary/1)]
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
