use std::convert::TryInto;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::*;
use firefly_rt::term::Term;

pub fn list_to_string(list: Term) -> Result<String, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok("".to_owned()),
        Term::Cons(cons) => cons
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
            .collect::<Result<String, NonNull<ErlangException>>>(),
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
