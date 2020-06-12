#[cfg(all(not(any(target_arch = "wasm32", feature = "runtime_minimal")), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(reverse/2)]
pub fn result(process: &Process, list: Term, tail: Term) -> exception::Result<Term> {
    match list.decode()? {
        TypedTerm::Nil => Ok(tail),
        TypedTerm::List(cons) => {
            let mut reversed = tail;

            for result in cons.into_iter() {
                match result {
                    Ok(element) => {
                        reversed = process.cons(element, reversed)?;
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
