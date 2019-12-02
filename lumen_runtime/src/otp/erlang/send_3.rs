// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::send::{self, send, Sent};

// `send(destination, message, [nosuspend])` is used in `gen.erl`, which is used by `gen_server.erl`
// See https://github.com/erlang/otp/blob/8f6d45ddc8b2b12376c252a30b267a822cad171a/lib/stdlib/src/gen.erl#L167
#[native_implemented_function(send/3)]
pub fn native(
    process: &Process,
    destination: Term,
    message: Term,
    options: Term,
) -> exception::Result<Term> {
    let send_options: send::Options = options.try_into().map_err(|_| badarg!(process))?;

    send(destination, message, send_options, process)
        .map(|sent| match sent {
            Sent::Sent => "ok",
            Sent::ConnectRequired => "noconnect",
            Sent::SuspendRequired => "nosuspend",
        })
        .map(Atom::str_to_term)
}
