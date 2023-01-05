use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `bor/2` infix operator.
#[native_implemented::function(erlang:bor/2)]
pub fn result(
    process: &Process,
    left_integer: Term,
    right_integer: Term,
) -> Result<Term, NonNull<ErlangException>> {
    bitwise_infix_operator!(left_integer, right_integer, process, bitor)
}
