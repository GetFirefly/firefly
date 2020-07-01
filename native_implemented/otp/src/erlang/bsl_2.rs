#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `bsl/2` infix operator.
#[native_implemented::function(erlang:bsl/2)]
pub fn result(process: &Process, integer: Term, shift: Term) -> exception::Result<Term> {
    bitshift_infix_operator!(integer, shift, process, <<, >>)
}
