#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `-/2` infix operator
#[native_implemented::function(erlang:-/2)]
pub fn result(
    process: &Process,
    minuend: Term,
    subtrahend: Term,
) -> Result<Term, NonNull<ErlangException>> {
    number_infix_operator!(minuend, subtrahend, process, checked_sub, -)
}
