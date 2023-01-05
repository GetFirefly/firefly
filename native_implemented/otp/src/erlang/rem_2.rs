#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `rem/2` infix operator.  Integer remainder.
#[native_implemented::function(erlang:rem/2)]
pub fn result(
    process: &Process,
    dividend: Term,
    divisor: Term,
) -> Result<Term, NonNull<ErlangException>> {
    integer_infix_operator!(dividend, divisor, process, %)
}
