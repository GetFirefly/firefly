#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:list_to_tuple/1)]
pub fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok(process.tuple_term_from_term_slice(&[])),
        Term::Cons(cons) => {
            let vec: Vec<Term> = cons
                .into_iter()
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| ImproperListError)
                .with_context(|| format!("list ({}) is improper", list))?;

            Ok(process.tuple_term_from_term_slice(&vec))
        }
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
