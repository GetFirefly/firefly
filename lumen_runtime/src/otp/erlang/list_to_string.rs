use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

pub fn list_to_string(list: Term) -> exception::Result<String> {
    match list.decode()? {
        TypedTerm::Nil => Ok("".to_owned()),
        TypedTerm::List(cons) => cons
            .into_iter()
            .map(|result| match result {
                Ok(term) => {
                    let c: char = term.try_into().with_context(|| {
                        format!(
                            "string list ({}) element ({}) must be a unicode scalar value",
                            list, term
                        )
                    })?;

                    Ok(c)
                }
                Err(_) => Err(ImproperListError)
                    .context(format!("list ({}) is improper", list))
                    .map_err(From::from),
            })
            .collect::<exception::Result<String>>(),
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
