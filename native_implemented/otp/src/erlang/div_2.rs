use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `div/2` infix operator.  Integer division.
#[native_implemented::function(erlang:div/2)]
pub fn result(
    process: &Process,
    dividend: Term,
    divisor: Term,
) -> Result<Term, NonNull<ErlangException>> {
    integer_infix_operator!(dividend, divisor, process, /)
}
