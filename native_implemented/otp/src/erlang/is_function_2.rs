#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

use crate::runtime::context::*;

#[native_implemented::function(erlang:is_function/2)]
fn result(term: Term, arity: Term) -> Result<Term, NonNull<ErlangException>> {
    let arity_arity = term_try_into_arity(arity)?;

    Ok(term.is_function_with_arity(arity_arity).into())
}
