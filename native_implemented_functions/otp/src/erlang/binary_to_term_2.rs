// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::u8;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

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

#[native_implemented_function(binary_to_term/2)]
pub fn result(process: &Process, binary: Term, options: Term) -> exception::Result<Term> {
    let options: Options = options.try_into()?;

    match binary.decode()? {
        TypedTerm::HeapBinary(heap_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, heap_binary.as_bytes())
        }
        TypedTerm::MatchContext(match_context) => {
            maybe_aligned_maybe_binary_try_into_term!(process, &options, binary, match_context)
        }
        TypedTerm::ProcBin(process_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, process_binary.as_bytes())
        }
        TypedTerm::SubBinary(subbinary) => {
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
) -> exception::Result<Term> {
    let after_version_bytes = version::check(bytes)?;
    let (term, after_term_bytes) =
        term::decode_tagged(process, options.existing, after_version_bytes)?;

    if options.used {
        let used_byte_len = bytes.len() - after_term_bytes.len();
        let used = process.integer(used_byte_len)?;

        process
            .tuple_from_slice(&[term, used])
            .map_err(|alloc| alloc.into())
    } else {
        Ok(term)
    }
}
