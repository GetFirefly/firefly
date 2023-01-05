#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

use crate::erlang::is_record;

#[native_implemented::function(erlang:is_record/3)]
pub fn result(term: Term, record_tag: Term, size: Term) -> Result<Term, NonNull<ErlangException>> {
    is_record(term, record_tag, Some(size))
}
