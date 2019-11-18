// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Term, TypedTerm};
use lumen_runtime_macros::native_implemented_function;

use crate::otp;

/// Returns a binary that is made from the integers and binaries given in iolist
#[native_implemented_function(iolist_to_iovec/1)]
pub fn native(process: &Process, iolist_or_binary: Term) -> exception::Result {
    let iovec: Vec<Term> = if iolist_or_binary.is_binary() {
        vec![iolist_or_binary]
    } else {
        match iolist_or_binary.to_typed_term().unwrap() {
            TypedTerm::List(boxed_cons) => {
                let mut binaries: Vec<Term> = Vec::new();

                for item in boxed_cons.into_iter() {
                    let term: Term = match item {
                        Ok(term) => term,
                        _ => return Err(badarg!().into()),
                    };

                    if term.is_binary() {
                        binaries.push(term);
                    } else {
                        match otp::erlang::list_to_binary_1::native(process, term) {
                            Ok(term) => binaries.push(term),
                            _ => return Err(badarg!().into()),
                        }
                    }
                }

                binaries
            }
            _ => return Err(badarg!().into()),
        }
    };

    Ok(process.list_from_slice(&iovec).unwrap())
}
