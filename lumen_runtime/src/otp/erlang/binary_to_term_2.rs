// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::u8;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::binary::to_term::Options;
use crate::distribution::external_term_format::{term, VERSION_NUMBER};

macro_rules! maybe_aligned_maybe_binary_try_into_term {
    ($process:expr, $options:expr, $ident:expr) => {
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
            Err(badarg!($process).into())
        }
    };
}

#[native_implemented_function(binary_to_term/2)]
pub fn native(process: &Process, binary: Term, options: Term) -> exception::Result<Term> {
    let options: Options = options.try_into().map_err(|_| badarg!(process))?;

    match binary.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, heap_binary.as_bytes())
        }
        TypedTerm::MatchContext(match_context) => {
            maybe_aligned_maybe_binary_try_into_term!(process, &options, match_context)
        }
        TypedTerm::ProcBin(process_binary) => {
            versioned_tagged_bytes_try_into_term(process, &options, process_binary.as_bytes())
        }
        TypedTerm::SubBinary(subbinary) => {
            maybe_aligned_maybe_binary_try_into_term!(process, &options, subbinary)
        }
        _ => Err(badarg!(process).into()),
    }
}

fn versioned_tagged_bytes_try_into_term(
    process: &Process,
    options: &Options,
    bytes: &[u8],
) -> exception::Result<Term> {
    if 1 <= bytes.len() && bytes[0] == VERSION_NUMBER {
        let (term, after_term_bytes) = term::decode_tagged(process, options.existing, &bytes[1..])?;

        if options.used {
            let used_byte_len = bytes.len() - after_term_bytes.len();
            let used = process.integer(used_byte_len)?;

            process
                .tuple_from_slice(&[term, used])
                .map_err(|alloc| alloc.into())
        } else {
            Ok(term)
        }
    } else {
        Err(badarg!(process).into())
    }
}
