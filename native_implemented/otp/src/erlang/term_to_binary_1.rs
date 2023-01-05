#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::term_to_binary::term_to_binary;

#[native_implemented::function(erlang:term_to_binary/1)]
pub fn result(process: &Process, term: Term) -> Term {
    term_to_binary(process, term, Default::default())
}
