#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `*/2` infix operator
#[native_implemented::function(erlang:*/2)]
pub fn result(process: &Process, multiplier: Term, multiplicand: Term) -> exception::Result<Term> {
    number_infix_operator!(multiplier, multiplicand, process, checked_mul, *)
}
