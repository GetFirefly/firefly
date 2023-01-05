use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `+/2` infix operator
#[native_implemented::function(erlang:+/2)]
pub fn result(
    process: &Process,
    augend: Term,
    addend: Term,
) -> Result<Term, NonNull<ErlangException>> {
    number_infix_operator!(augend, addend, process, checked_add, +)
}
