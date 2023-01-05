#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang;

/// Returns a binary that is made from the integers and binaries given in iolist
#[native_implemented::function(erlang:iolist_to_binary/1)]
pub fn result(process: &Process, iolist_or_binary: Term) -> Result<Term, NonNull<ErlangException>> {
    erlang::list_to_binary_1::result(
        process,
        if iolist_or_binary.is_binary() {
            process.list_from_slice(&[iolist_or_binary]).unwrap()
        } else {
            iolist_or_binary
        },
    )
}
