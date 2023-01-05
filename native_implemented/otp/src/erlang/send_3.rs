#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Atom, Term};

use crate::runtime::send::{self, send, Sent};

// `send(destination, message, [nosuspend])` is used in `gen.erl`, which is used by `gen_server.erl`
// See https://github.com/erlang/otp/blob/8f6d45ddc8b2b12376c252a30b267a822cad171a/lib/stdlib/src/gen.erl#L167
#[native_implemented::function(erlang:send/3)]
pub fn result(
    process: &Process,
    destination: Term,
    message: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let send_options: send::Options = options.try_into()?;

    send(destination, message, send_options, process)
        .map(|sent| match sent {
            Sent::Sent => "ok",
            Sent::ConnectRequired => "noconnect",
            Sent::SuspendRequired => "nosuspend",
        })
        .map(Atom::str_to_term)
        .map_err(From::from)
}
