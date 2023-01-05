use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::{Term, TypeError};

use crate::runtime::context::term_is_not_type;

pub fn arguments_term_to_vec(arguments: Term) -> Result<Vec<Term>, NonNull<ErlangException>> {
    let mut argument_vec: Vec<Term> = Vec::new();

    match arguments {
        Term::Cons(arguments_boxed_cons) => {
            for result in arguments_boxed_cons.iter() {
                match result {
                    Ok(element) => argument_vec.push(element),
                    Err(_) => {
                        return Err(anyhow!(ImproperListError))
                            .context(term_is_not_type(
                                "arguments",
                                arguments,
                                "a proper list list",
                            ))
                            .map_err(From::from)
                    }
                }
            }

            Ok(argument_vec)
        }
        Term::Nil => Ok(argument_vec),
        _ => Err(TypeError)
            .context(term_is_not_type(
                "arguments",
                arguments,
                "a proper list list",
            ))
            .map_err(From::from),
    }
}
