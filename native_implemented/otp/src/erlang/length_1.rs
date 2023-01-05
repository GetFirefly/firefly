#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(erlang:length/1)]
pub fn result(process: &Process, list: Term) -> Result<Term, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok(0.into()),
        Term::Cons(cons) => match cons.count() {
            Some(count) => Ok(process.integer(count).unwrap()),
            None => Err(ImproperListError).context(format!("list ({}) is improper", list)),
        },
        _ => Err(TypeError).context(format!("list ({}) is not a list", list)),
    }
    .map_err(From::from)
}
