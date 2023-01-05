#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::ptr::NonNull;
use std::u8;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::*;
use firefly_rt::term::{Term, TypeError};

use crate::binary::to_term::Options;
use crate::runtime::distribution::external_term_format::{term, version};

macro_rules! maybe_aligned_maybe_binary_try_into_term {
    ($process:expr, $options:expr, $binary:expr, $ident:expr) => {
        if $ident.is_binary() {
            if $ident.is_aligned() {
                versioned_tagged_bytes_try_into_term($process, $options, unsafe {
                    $ident.as_bytes_unchecked()
                })
            } else {
                let byte_vec: Vec<u8> = $ident.full_byte_iter().collect();
                versioned_tagged_bytes_try_into_term($process, $options, &byte_vec)
            }
        } else {
            Err(NotABinary)
                .context(format!(
                    "binary ({}) is a bitstring, but not a binary",
                    $binary
                ))
                .map_err(From::from)
        }
    };
}

#[native_implemented::function(erlang:binary_to_term/2)]
pub fn result(
    process: &Process,
    binary: Term,
    options: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let options: Options = options.try_into()?;

    match binary {
        Term::HeapBinary(heap_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, heap_binary.as_bytes())
        }
        Term::RcBinary(process_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, process_binary.as_bytes())
        }
        Term::RefBinary(subbinary) => {
            maybe_aligned_maybe_binary_try_into_term!(process, &options, binary, subbinary)
        }
        _ => Err(TypeError)
            .context(format!("binary ({}) is not a binary", binary))
            .map_err(From::from),
    }
}

fn versioned_tagged_bytes_try_into_term(
    process: &Process,
    options: &Options,
    bytes: &[u8],
) -> Result<Term, NonNull<ErlangException>> {
    let after_version_bytes = version::check(bytes)?;
    let (term, after_term_bytes) =
        term::decode_tagged(process, options.existing, after_version_bytes)?;

    let final_term = if options.used {
        let used_byte_len = bytes.len() - after_term_bytes.len();
        let used = process.integer(used_byte_len).unwrap();

        process.tuple_term_from_term_slice(&[term, used])
    } else {
        term
    };

    Ok(final_term)
}
