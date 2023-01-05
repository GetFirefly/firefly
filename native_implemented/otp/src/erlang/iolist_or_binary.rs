use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::context::{r#type, term_is_not_type};

pub fn element_not_a_binary_context(iolist_or_binary: Term, element: Term) -> String {
    format!(
        "iolist_or_binary ({}) element ({}) is a bitstring, but not a binary",
        iolist_or_binary, element
    )
}

pub fn element_type_context(iolist_or_binary: Term, element: Term) -> String {
    format!(
        "iolist_or_binary ({}) element ({}) is not a byte, binary, or nested iolist ({})",
        iolist_or_binary,
        element,
        r#type::IOLIST
    )
}

pub fn result(
    process: &Process,
    iolist_or_binary: Term,
    try_into: fn(&Process, Term) -> Result<Term, NonNull<ErlangException>>,
) -> Result<Term, NonNull<ErlangException>> {
    match iolist_or_binary {
        Term::Nil
        | Term::Cons(_)
        | Term::ConstantBinary(_)
        | Term::HeapBinary(_)
        | Term::RcBinary(_)
        | Term::RefBinary(_) => try_into(process, iolist_or_binary),
        _ => Err(TypeError)
            .context(term_is_not_type(
                "iolist_or_binary",
                iolist_or_binary,
                &format!("an iolist ({}) or binary", r#type::IOLIST),
            ))
            .map_err(From::from),
    }
}

pub fn to_binary(
    process: &Process,
    name: &'static str,
    value: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let mut byte_vec: Vec<u8> = Vec::new();
    let mut stack: Vec<Term> = vec![value];

    while let Some(top) = stack.pop() {
        match top {
            Term::Int(small_integer) => {
                let top_byte = small_integer
                    .try_into()
                    .with_context(|| element_context(name, value, top))?;

                byte_vec.push(top_byte);
            }
            Term::Nil => (),
            Term::Cons(non_null_cons) => {
                let cons = unsafe { non_null_cons.as_ref() };
                // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                // binary() | []) means that `byte()` isn't allowed
                // for `tail`s unlike `head`.

                let tail = cons.tail();
                let result_u8: Result<u8, _> = tail.try_into();

                match result_u8 {
                    Ok(_) => {
                        return Err(TypeError)
                            .context(format!(
                                "{} ({}) tail ({}) cannot be a byte",
                                name, value, tail
                            ))
                            .map_err(From::from)
                    }
                    Err(_) => stack.push(tail),
                };

                stack.push(cons.head());
            }
            Term::HeapBinary(heap_binary) => {
                byte_vec.extend_from_slice(heap_binary.as_bytes());
            }
            Term::RefBinary(subbinary) => {
                if subbinary.is_binary() {
                    if subbinary.is_aligned() {
                        byte_vec.extend(unsafe { subbinary.as_bytes_unchecked() });
                    } else {
                        byte_vec.extend(subbinary.full_byte_iter());
                    }
                } else {
                    return Err(NotABinary)
                        .context(element_context(name, value, top))
                        .map_err(From::from);
                }
            }
            Term::RcBinary(procbin) => {
                byte_vec.extend_from_slice(procbin.as_bytes());
            }
            _ => {
                return Err(TypeError)
                    .context(element_context(name, value, top))
                    .map_err(From::from)
            }
        }
    }

    Ok(process.binary_from_bytes(byte_vec.as_slice()))
}

fn element_context(name: &'static str, value: Term, element: Term) -> String {
    format!(
        "{} ({}) element ({}) is not a byte, binary, or nested iolist",
        name, value, element
    )
}
