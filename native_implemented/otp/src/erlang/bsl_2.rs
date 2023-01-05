use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `bsl/2` infix operator.
#[native_implemented::function(erlang:bsl/2)]
pub fn result(
    process: &Process,
    integer: Term,
    shift: Term,
) -> Result<Term, NonNull<ErlangException>> {
    bitshift_infix_operator!(integer, shift, process, <<, >>)
}
