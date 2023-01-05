#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::erlang::iolist_or_binary::{self, *};

/// Returns the size, in bytes, of the binary that would be result from iolist_to_binary/1
#[native_implemented::function(erlang:iolist_size/1)]
pub fn result(process: &Process, iolist_or_binary: Term) -> Result<Term, NonNull<ErlangException>> {
    iolist_or_binary::result(process, iolist_or_binary, iolist_or_binary_size)
}

fn iolist_or_binary_size(
    process: &Process,
    iolist_or_binary: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let mut size: usize = 0;
    let mut stack: Vec<Term> = vec![iolist_or_binary];

    while let Some(top) = stack.pop() {
        match top {
            Term::Int(small_integer) => {
                let _: u8 = small_integer
                    .try_into()
                    .with_context(|| element_type_context(iolist_or_binary, top))?;

                size += 1;
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
                                "iolist_or_binary ({}) tail ({}) cannot be a byte",
                                iolist_or_binary, tail
                            ))
                            .map_err(From::from)
                    }
                    Err(_) => stack.push(tail),
                };

                stack.push(cons.head());
            }
            Term::HeapBinary(heap_binary) => size += heap_binary.full_byte_len(),
            Term::RcBinary(proc_bin) => size += proc_bin.total_byte_len(),
            Term::RefBinary(subbinary) => {
                if subbinary.is_binary() {
                    size += subbinary.full_byte_len();
                } else {
                    return Err(NotABinary)
                        .context(element_not_a_binary_context(iolist_or_binary, top))
                        .map_err(From::from);
                }
            }
            _ => {
                return Err(TypeError)
                    .context(element_type_context(iolist_or_binary, top))
                    .map_err(From::from);
            }
        }
    }

    Ok(process.integer(size).unwrap())
}
