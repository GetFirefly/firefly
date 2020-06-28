#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::term_try_into_one_based_index;

#[native_implemented::function(keyfind/3)]
pub fn result(key: Term, index: Term, tuple_list: Term) -> exception::Result<Term> {
    let index = term_try_into_one_based_index(index)?;

    match tuple_list.decode()? {
        TypedTerm::Nil => Ok(false.into()),
        TypedTerm::List(cons) => match cons.keyfind(index, key)? {
            Some(found) => Ok(found),
            None => Ok(false.into()),
        },
        _ => Err(ImproperListError)
            .context(format!("tuple_list ({}) is not a proper list", tuple_list))
            .map_err(From::from),
    }
}
