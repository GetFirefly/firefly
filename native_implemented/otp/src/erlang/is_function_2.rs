#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::*;

#[native_implemented::function(erlang:is_function/2)]
fn result(term: Term, arity: Term) -> exception::Result<Term> {
    let arity_arity = term_try_into_arity(arity)?;

    Ok(term.decode()?.is_function_with_arity(arity_arity).into())
}
