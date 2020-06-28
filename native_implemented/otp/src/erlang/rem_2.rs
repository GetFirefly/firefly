#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `rem/2` infix operator.  Integer remainder.
#[native_implemented::function(rem/2)]
pub fn result(process: &Process, dividend: Term, divisor: Term) -> exception::Result<Term> {
    integer_infix_operator!(dividend, divisor, process, %)
}
