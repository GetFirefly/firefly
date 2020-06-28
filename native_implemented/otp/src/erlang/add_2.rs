#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

/// `+/2` infix operator
#[native_implemented::function(+/2)]
pub fn result(process: &Process, augend: Term, addend: Term) -> exception::Result<Term> {
    number_infix_operator!(augend, addend, process, checked_add, +)
}
