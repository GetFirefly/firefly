#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `*/2` infix operator
#[native_implemented::function(erlang:*/2)]
pub fn result(
    process: &Process,
    multiplier: Term,
    multiplicand: Term,
) -> Result<Term, NonNull<ErlangException>> {
    number_infix_operator!(multiplier, multiplicand, process, checked_mul, *)
}
