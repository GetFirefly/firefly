use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::{Term, TypeError};

pub fn charlist_to_string(list: Term) -> Result<String, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok("".to_string()),
        Term::Cons(boxed_cons) => boxed_cons.try_into().map_err(From::from),
        _ => Err(TypeError)
            .context(format!("list ({}) is not a a list", list))
            .map_err(From::from),
    }
}
