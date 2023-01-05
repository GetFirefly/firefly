use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::function::module_loaded;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:module_loaded/1)]
pub fn result(module: Term) -> Result<Term, NonNull<ErlangException>> {
    let module_atom = term_try_into_atom!(module)?;

    Ok(module_loaded(module_atom).into())
}
