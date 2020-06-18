#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::*;

#[native_implemented::function(lists:keymember/3)]
pub fn result(key: Term, index: Term, tuple_list: Term) -> exception::Result<Term> {
    let index = term_try_into_one_based_index(index)?;

    match tuple_list.decode()? {
        TypedTerm::Nil => Ok(false.into()),
        TypedTerm::List(cons) => match cons.keyfind(index, key)? {
            Some(_) => Ok(true.into()),
            None => Ok(false.into()),
        },
        _ => Err(TypeError)
            .context(format!("tuple_list ({}) is not a proper list", tuple_list))
            .map_err(From::from),
    }
}
