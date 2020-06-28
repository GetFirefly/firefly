#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::Term;

use crate::erlang::is_record;

#[native_implemented::function(is_record/3)]
pub fn result(term: Term, record_tag: Term, size: Term) -> exception::Result<Term> {
    is_record(term, record_tag, Some(size))
}
