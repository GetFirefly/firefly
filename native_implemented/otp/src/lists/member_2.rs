#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::term::Term;

#[native_implemented::function(lists:member/2)]
pub fn result(element: Term, list: Term) -> Result<Term, NonNull<ErlangException>> {
    match list {
        Term::Nil => Ok(false.into()),
        Term::Cons(cons) => {
            for result in cons.into_iter() {
                match result {
                    Ok(term) => {
                        if term == element {
                            return Ok(true.into());
                        }
                    }
                    Err(_) => {
                        return Err(ImproperListError)
                            .context(format!("list ({}) is improper", list))
                            .map_err(From::from)
                    }
                };
            }

            Ok(false.into())
        }
        _ => Err(TypeError)
            .context(format!("list ({}) is not a list", list))
            .map_err(From::from),
    }
}
