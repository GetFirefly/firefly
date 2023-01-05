use std::ptr::NonNull;

use firefly_rt::error::ErlangException;
use firefly_rt::term::{Cons, Term};

#[native_implemented::function(erlang:hd/1)]
pub fn result(list: Term) -> Result<Term, NonNull<ErlangException>> {
    let cons: Boxed<Cons> = term_try_into_non_empty_list!(list)?;

    Ok(cons.head)
}
