use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::{Term, TypedTerm};

pub fn list_to_string(list: Term) -> Result<String, Exception> {
    match list.to_typed_term().unwrap() {
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
            .collect::<Result<String, Exception>>(),
        _ => Err(badarg!().into()),
    }
}
