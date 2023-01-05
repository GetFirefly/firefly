use std::ptr::NonNull;
use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{ImproperList, Term, TypeError};

/// `++/2`
#[native_implemented::function(erlang:++/2)]
pub fn result(process: &Process, list: Term, term: Term) -> Result<Term, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok(term),
        Term::Cons(cons) => match cons
            .into_iter()
            .collect::<std::result::Result<Vec<Term>, _>>()
        {
            Ok(vec) => Ok(process.improper_list_from_slice(&vec, term)),
            Err(ImproperList { .. }) => Err(ImproperListError)
                .context(format!("list ({}) is improper", list))
                .map_err(From::from),
        },
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
