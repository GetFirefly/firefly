#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::binary_to_term_2;

#[native_implemented::function(erlang:binary_to_term/1)]
pub fn result(process: &Process, binary: Term) -> Result<Term, NonNull<ErlangException>> {
    binary_to_term_2::result(process, binary, Term::Nil)
}
