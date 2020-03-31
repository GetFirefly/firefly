use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

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

pub fn native(
    process: &Process,
    iolist_or_binary: Term,
    try_into: fn(&Process, Term) -> exception::Result<Term>,
) -> exception::Result<Term> {
    match iolist_or_binary.decode()? {
        TypedTerm::Nil
        | TypedTerm::List(_)
        | TypedTerm::BinaryLiteral(_)
        | TypedTerm::HeapBinary(_)
        | TypedTerm::MatchContext(_)
        | TypedTerm::ProcBin(_)
        | TypedTerm::SubBinary(_) => try_into(process, iolist_or_binary),
        _ => Err(TypeError)
            .context(term_is_not_type(
                "iolist_or_binary",
                iolist_or_binary,
                &format!("an iolist ({}) or binary", r#type::IOLIST),
            ))
            .map_err(From::from),
    }
}

pub fn to_binary(process: &Process, name: &'static str, value: Term) -> exception::Result<Term> {
    let mut byte_vec: Vec<u8> = Vec::new();
    let mut stack: Vec<Term> = vec![value];

    while let Some(top) = stack.pop() {
        match top.decode()? {
            TypedTerm::SmallInteger(small_integer) => {
                let top_byte = small_integer
                    .try_into()
                    .with_context(|| element_context(name, value, top))?;

                byte_vec.push(top_byte);
            }
            TypedTerm::Nil => (),
            TypedTerm::List(boxed_cons) => {
                // @type iolist :: maybe_improper_list(byte() | binary() | iolist(),
                // binary() | []) means that `byte()` isn't allowed
                // for `tail`s unlike `head`.

                let tail = boxed_cons.tail;
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

                stack.push(boxed_cons.head);
            }
            TypedTerm::HeapBinary(heap_binary) => {
                byte_vec.extend_from_slice(heap_binary.as_bytes());
            }
            TypedTerm::SubBinary(subbinary) => {
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
            TypedTerm::ProcBin(procbin) => {
                byte_vec.extend_from_slice(procbin.as_bytes());
            }
            _ => {
                return Err(TypeError)
                    .context(element_context(name, value, top))
                    .map_err(From::from)
            }
        }
    }

    Ok(process.binary_from_bytes(byte_vec.as_slice()).unwrap())
}

fn element_context(name: &'static str, value: Term, element: Term) -> String {
    format!(
        "{} ({}) element ({}) is not a byte, binary, or nested iolist",
        name, value, element
    )
}
