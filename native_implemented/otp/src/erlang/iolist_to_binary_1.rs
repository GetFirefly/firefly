#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::erlang;

/// Returns a binary that is made from the integers and binaries given in iolist
#[native_implemented::function(iolist_to_binary/1)]
pub fn result(process: &Process, iolist_or_binary: Term) -> exception::Result<Term> {
    erlang::list_to_binary_1::result(
        process,
        if iolist_or_binary.is_binary() {
            process.list_from_slice(&[iolist_or_binary]).unwrap()
        } else {
            iolist_or_binary
        },
    )
}
