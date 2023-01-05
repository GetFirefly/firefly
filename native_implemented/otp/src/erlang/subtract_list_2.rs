#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

/// `--/2`
#[native_implemented::function(erlang:--/2)]
pub fn result(
    process: &Process,
    minuend: Term,
    subtrahend: Term,
) -> Result<Term, NonNull<ErlangException>> {
    match minuend {
        Term::Nil => match subtrahend {
            Term::Nil => Ok(minuend),
            Term::Cons(subtrahend_cons) => {
                if subtrahend_cons.is_proper() {
                    Ok(minuend)
                } else {
                    Err(ImproperListError).context(is_not_a_proper_list("subtrahend", subtrahend))
                }
            }
            _ => Err(TypeError).context(is_not_a_proper_list("subtrahend", subtrahend)),
        },
        Term::Cons(minuend_cons) => match subtrahend {
            Term::Nil => {
                if minuend_cons.is_proper() {
                    Ok(minuend)
                } else {
                    Err(ImproperListError)
                        .context(is_not_a_proper_list("minuend", minuend))
                        .map_err(From::from)
                }
            }
            Term::Cons(subtrahend_cons) => {
                match minuend_cons
                    .into_iter()
                    .collect::<std::result::Result<Vec<Term>, _>>()
                {
                    Ok(mut minuend_vec) => {
                        for result in subtrahend_cons.into_iter() {
                            match result {
                                Ok(subtrahend_element) => {
                                    if let Some(index) =
                                        minuend_vec.iter().position(|minuend_element| {
                                            *minuend_element == subtrahend_element
                                        })
                                    {
                                        minuend_vec.remove(index);
                                    }
                                }
                                Err(ImproperList { .. }) => {
                                    return Err(ImproperListError)
                                        .context(is_not_a_proper_list("subtrahend", subtrahend))
                                        .map_err(From::from)
                                }
                            };
                        }

                        Ok(process.list_from_slice(&minuend_vec)).unwrap()
                    }
                    Err(ImproperList { .. }) => {
                        Err(ImproperListError).context(is_not_a_proper_list("minuend", minuend))
                    }
                }
            }
            _ => Err(TypeError).context(is_not_a_proper_list("subtrahend", subtrahend)),
        },
        _ => match subtrahend {
            Term::Nil | Term::Cons(_) => {
                Err(TypeError).context(is_not_a_proper_list("minuend", minuend))
            }
            _ => Err(TypeError).context(format!(
                "neither minuend ({}) nor subtrahend ({}) are a proper list",
                minuend, subtrahend
            )),
        },
    }
    .map_err(From::from)
}

fn is_not_a_proper_list(name: &str, value: Term) -> String {
    format!("{} ({}) is not a proper list", name, value)
}
