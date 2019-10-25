#[cfg(test)]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

use crate::otp::erlang::term_to_binary::term_to_binary;

#[native_implemented_function(term_to_binary/1)]
pub fn native(process: &Process, term: Term) -> exception::Result {
    term_to_binary(process, term, Default::default())
}
