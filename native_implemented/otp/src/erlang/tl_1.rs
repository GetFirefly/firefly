use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(erlang:tl/1)]
pub fn result(list: Term) -> exception::Result<Term> {
    let cons: Boxed<Cons> = term_try_into_non_empty_list!(list)?;

    Ok(cons.tail)
}
