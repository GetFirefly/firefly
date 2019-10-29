// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::send::{send, Sent};

#[native_implemented_function(send/2)]
pub fn native(process: &Process, destination: Term, message: Term) -> exception::Result<Term> {
    send(destination, message, Default::default(), process).map(|sent| match sent {
        Sent::Sent => message,
        _ => unreachable!(),
    })
}
