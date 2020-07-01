#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(lists:member/2)]
pub fn result(element: Term, list: Term) -> exception::Result<Term> {
    match list.decode()? {
        TypedTerm::Nil => Ok(false.into()),
        TypedTerm::List(cons) => {
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
