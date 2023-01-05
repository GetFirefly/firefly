#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::*;
use firefly_rt::term::Term;

use crate::runtime::context::term_try_into_one_based_index;

#[native_implemented::function(lists:keyfind/3)]
pub fn result(key: Term, index: Term, tuple_list: Term) -> Result<Term, NonNull<ErlangException>> {
    let index = term_try_into_one_based_index(index)?;

    match tuple_list {
        Term::Nil => Ok(false.into()),
        Term::Cons(cons) => match cons.keyfind(index, key)? {
            Some(found) => Ok(found),
            None => Ok(false.into()),
        },
        _ => Err(ImproperListError)
            .context(format!("tuple_list ({}) is not a proper list", tuple_list))
            .map_err(From::from),
    }
}
