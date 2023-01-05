#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::send::{send, Sent};

#[native_implemented::function(erlang:send/2)]
pub fn result(
    process: &Process,
    destination: Term,
    message: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let sent = send(destination, message, Default::default(), process)?;

    match sent {
        Sent::Sent => Ok(message),
        _ => unreachable!(),
    }
}
