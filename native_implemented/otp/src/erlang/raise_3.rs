#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::*;
use num_bigint::BigInt;

use firefly_rt::backtrace::Trace;
use firefly_rt::error::ErlangException;
use firefly_rt::term::{Term, Tuple};

#[native_implemented::function(erlang:raise/3)]
pub fn result(
    class: Term,
    reason: Term,
    stacktrace: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let class_class: exception::Class = class.try_into()?;
    let trace =
        trace_try_from_term(stacktrace).with_context(|| format!("stacktrace ({})", stacktrace))?;

    Err(raise(
        class_class,
        reason,
        trace,
        Some(anyhow!("explicit raise from Erlang").into()),
    )
    .into())
}

fn trace_try_from_term(stacktrace: Term) -> anyhow::Result<Arc<Trace>> {
    match stacktrace {
        Term::Nil => Ok(Trace::from_term(stacktrace)),
        Term::Cons(cons) => {
            for (index, result) in cons.iter().enumerate() {
                match result {
                    Ok(element) => item_try_from_term(element).with_context(|| {
                        format!("at index ({}) has element ({})", index, element)
                    })?,
                    Err(_) => return Err(anyhow!("is not a proper list")),
                }
            }

            Ok(Trace::from_term(stacktrace))
        }
        _ => Err(anyhow!("is not a list")),
    }
}

fn item_try_from_term(term: Term) -> anyhow::Result<()> {
    let tuple = term.try_into().map_err(|_| anyhow!("is not a tuple"))?;

    item_try_from_tuple(tuple)
}

fn item_try_from_tuple(tuple: NonNull<Tuple>) -> anyhow::Result<()> {
    match tuple.len() {
        2 => item_try_from_2_tuple(tuple).context("2-tuple"),
        3 => item_try_from_3_tuple(tuple).context("3-tuple"),
        4 => item_try_from_4_tuple(tuple).context("4-tuple"),
        _ => Err(anyhow!("is not one of the supported formats:\n1. `{Function, Args}`\n2. `{Module, Function, Arity | Args}`\n3. `{Function, Args, Location}`\n4. `{Module, Function, Arity | Args, Location}`")),
    }
}

// {function, args}
// https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1107-L1114
fn item_try_from_2_tuple(tuple: NonNull<Tuple>) -> anyhow::Result<()> {
    let fun = tuple[0];

    if fun.is_closure() {
        Ok(())
    } else {
        Err(anyhow!("`Fun` ({}) is not a function", fun)).context("is not format `{{Fun, Args}}`")
    }
}

// https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1115-L1128
fn item_try_from_3_tuple(tuple: NonNull<Tuple>) -> anyhow::Result<()> {
    let first_element = tuple[0];

    match first_element {
        // {M, F, arity | args}
        Term::Atom(_) => {
            const FORMAT: &str = "is not format `{Module, Function, Arity | Args}`";

            let function = tuple[1];
            let _: Atom = function.try_into().map_err(|_| anyhow!("`Function` ({}) is not an atom", function)).context(FORMAT)?;

            arity_or_arguments_try_from_term(tuple[2]).context(FORMAT)?;

            Ok(())
        },
        // {function, args, location}
        Term::Closure(_) => {
            location_try_from_term(tuple[2]).context("is not format `{Function, Args, Location}`")
        },
        _ => Err(anyhow!("is not one of the supported formats:\n1. {Module, Function, Arity | Args}\n2. {Function, Args, Location}")),
    }
}

fn arity_or_arguments_try_from_term(term: Term) -> anyhow::Result<()> {
    match term {
        // args
        Term::Nil | Term::Cons(_) => Ok(()),
        // arity
        Term::Int(small_integer) => {
            let arity: isize = small_integer.into();

            if 0 <= arity {
                Ok(())
            } else {
                Err(anyhow!("`Arity` ({}) is not 0 or greater", term))
            }
        }
        // arity
        Term::BigInt(big_integer) => {
            let big_int = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &0.into();

            if zero_big_int <= big_int {
                Ok(())
            } else {
                Err(anyhow!("`Arity` ({}) is not 0 or greater", term))
            }
        }
        _ => Err(anyhow!("`Arity | Args` ({}) is not a number or list", term)),
    }
}

// https://github.com/erlang/otp/blob/b51f61b5f32a28737d0b03a29f19f48f38e4db19/erts/emulator/beam/bif.c#L1129-L1134
fn item_try_from_4_tuple(tuple: NonNull<Tuple>) -> anyhow::Result<()> {
    const FORMAT: &str = "is not format `{Module, Function, Arity | Args, Location}`";

    // {M, F, arity | args, location}
    let module = tuple[0];
    term_try_into_atom!(module).context(FORMAT)?;

    let function = tuple[1];
    term_try_into_atom!(function).context(FORMAT)?;

    arity_or_arguments_try_from_term(tuple[2]).context(FORMAT)?;
    location_try_from_term(tuple[3]).context(FORMAT)?;

    Ok(())
}

fn location_try_from_term(term: Term) -> anyhow::Result<()> {
    match term {
        Term::Nil => Ok(()),
        Term::Cons(cons) => {
            for (index, result) in cons.iter().enumerate() {
                match result {
                    Ok(element) => location_keyword_pair_from_term(element).with_context(|| {
                        format!(
                            "location ({}) at index ({}) has element ({})",
                            term, index, element
                        )
                    })?,
                    Err(_) => return Err(anyhow!("location ({}) is not a proper list", term)),
                }
            }

            Ok(())
        }
        _ => Err(anyhow!("location ({}) is not a list", term)),
    }
}

fn location_keyword_pair_from_term(keyword_pair: Term) -> anyhow::Result<()> {
    let tuple = term_try_into_tuple!(keyword_pair)?;

    location_keyword_pair_from_tuple(tuple)
}

fn location_keyword_pair_from_tuple(keyword_pair: NonNull<Tuple>) -> anyhow::Result<()> {
    if keyword_pair.len() == 2 {
        location_keyword_pair_from_2_tuple(keyword_pair)
            .with_context(|| format!("is not a keyword pair"))
    } else {
        Err(anyhow!("is not a 2-tuple"))
    }
}

fn location_keyword_pair_from_2_tuple(keyword_pair: NonNull<Tuple>) -> anyhow::Result<()> {
    let key = keyword_pair[0];
    let key_atom = term_try_into_atom!(key)?;
    let value = keyword_pair[1];

    match key_atom.as_str() {
        "file" => file_try_from_term(value),
        "line" => line_try_from_term(value),
        _ => Err(anyhow!("key ({}) is neither `file` nor `line`", key)),
    }
}

fn file_try_from_term(term: Term) -> anyhow::Result<()> {
    match term {
        Term::Cons(cons) => {
            for (index, result) in cons.iter().enumerate() {
                match result {
                    Ok(element) => character_try_from_term(element)
                        .with_context(|| format!("at index ({})", index))?,
                    Err(_) => return Err(anyhow!("file ({}) is an improper list")),
                };
            }

            Ok(())
        }
        _ => Err(anyhow!("file ({}) is not a non-empty list", term)),
    }
}

fn character_try_from_term(term: Term) -> anyhow::Result<char> {
    term.try_into()
        .context("is not a valid character in an Erlang string")
}

fn line_try_from_term(term: Term) -> anyhow::Result<()> {
    match term {
        Term::Int(small_integer) => {
            if 0_isize < small_integer.into() {
                Ok(())
            } else {
                Err(anyhow!("line ({}) is not 1 or greater", term))
            }
        }
        Term::BigInt(big_integer) => {
            let big_int = big_integer.as_ref().into();
            let zero_big_int: &BigInt = &0.into();

            if zero_big_int < big_int {
                Ok(())
            } else {
                Err(anyhow!("line ({}) is not 1 or greater", term))
            }
        }
        _ => Err(anyhow!("line ({}) is not a number", term)),
    }
}
