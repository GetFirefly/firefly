#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `band/2` infix operator.
#[native_implemented::function(band/2)]
pub fn result(
    process: &Process,
    left_integer: Term,
    right_integer: Term,
) -> exception::Result<Term> {
    bitwise_infix_operator!(left_integer, right_integer, process, bitand)
}
