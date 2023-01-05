#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

#[native_implemented::function(lists:reverse/2)]
pub fn result(process: &Process, list: Term, tail: Term) -> Result<Term, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok(tail),
        Term::Cons(cons) => {
            let mut reversed = tail;

            for result in cons.into_iter() {
                match result {
                    Ok(element) => {
                        reversed = process.cons(element, reversed);
                    }
                    Err(_) => {
                        return Err(ImproperListError)
                            .context(format!("list ({}) is not a proper list", list))
                            .map_err(From::from)
                    }
                }
            }

            Ok(reversed)
        }
        _ => Err(TypeError)
            .context(format!("list ({}) is not a proper list", list))
            .map_err(From::from),
    }
}
