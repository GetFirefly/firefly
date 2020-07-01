#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::send::{send, Sent};

#[native_implemented::function(erlang:send/2)]
pub fn result(process: &Process, destination: Term, message: Term) -> exception::Result<Term> {
    let sent = send(destination, message, Default::default(), process)?;

    match sent {
        Sent::Sent => Ok(message),
        _ => unreachable!(),
    }
}
