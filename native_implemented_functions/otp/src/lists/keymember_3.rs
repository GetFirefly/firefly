// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(keymember/3)]
pub fn native(key: Term, index: Term, tuple_list: Term) -> exception::Result<Term> {
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
