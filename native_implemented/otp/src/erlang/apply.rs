use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::context::term_is_not_type;

pub fn arguments_term_to_vec(arguments: Term) -> exception::Result<Vec<Term>> {
    let mut argument_vec: Vec<Term> = Vec::new();

    match arguments.decode().unwrap() {
        TypedTerm::List(arguments_boxed_cons) => {
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
        TypedTerm::Nil => Ok(argument_vec),
        _ => Err(TypeError)
            .context(term_is_not_type(
                "arguments",
                arguments,
                "a proper list list",
            ))
            .map_err(From::from),
    }
}
