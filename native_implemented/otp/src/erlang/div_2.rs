#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `div/2` infix operator.  Integer division.
#[native_implemented::function(erlang:div/2)]
pub fn result(process: &Process, dividend: Term, divisor: Term) -> exception::Result<Term> {
    integer_infix_operator!(dividend, divisor, process, /)
}
