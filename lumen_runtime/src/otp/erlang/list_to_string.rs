use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

pub fn list_to_string(list: Term) -> exception::Result<String> {
    match list.decode().unwrap() {
        TypedTerm::Nil => Ok("".to_owned()),
        TypedTerm::List(cons) => cons
            .into_iter()
            .map(|result| match result {
                Ok(term) => {
                    let c: char = term.try_into()?;

                    Ok(c)
                }
                Err(_) => Err(badarg!().into()),
            })
            .collect::<exception::Result<String>>(),
        _ => Err(badarg!().into()),
    }
}
