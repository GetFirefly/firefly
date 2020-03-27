use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

pub fn charlist_to_string(list: Term) -> exception::Result<String> {
    match list.decode()? {
        TypedTerm::Nil => Ok("".to_string()),
        TypedTerm::List(boxed_cons) => boxed_cons.try_into().map_err(From::from),
        _ => Err(TypeError)
            .context(format!("list ({}) is not a a list", list))
            .map_err(From::from),
    }
}
