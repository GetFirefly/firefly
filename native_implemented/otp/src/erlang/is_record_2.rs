#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::is_record;

#[native_implemented::function(is_record/2)]
pub fn result(term: Term, record_tag: Term) -> exception::Result<Term> {
    is_record(term, record_tag, None)
}
