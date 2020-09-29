use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

/// `++/2`
#[native_implemented::function(erlang:++/2)]
pub fn result(process: &Process, list: Term, term: Term) -> exception::Result<Term> {
    match list.decode().unwrap() {
        TypedTerm::Nil => Ok(term),
        TypedTerm::List(cons) => match cons
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
