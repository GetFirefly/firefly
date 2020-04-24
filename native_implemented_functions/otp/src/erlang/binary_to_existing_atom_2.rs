// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::string::Encoding;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

macro_rules! maybe_aligned_maybe_binary_to_atom {
    ($binary:ident, $maybe_aligned_maybe_binary:ident) => {
        if $maybe_aligned_maybe_binary.is_binary() {
            if $maybe_aligned_maybe_binary.is_aligned() {
                let bytes = unsafe { $maybe_aligned_maybe_binary.as_bytes_unchecked() };

                bytes_to_existing_atom($binary, bytes)
            } else {
                let byte_vec: Vec<u8> = $maybe_aligned_maybe_binary.full_byte_iter().collect();

                bytes_to_existing_atom($binary, &byte_vec)
            }
        } else {
            Err(NotABinary)
                .with_context(|| term_is_not_binary("bitstring", $binary))
                .map_err(From::from)
        }
    };
}

#[native_implemented_function(binary_to_existing_atom/2)]
pub fn result(binary: Term, encoding: Term) -> exception::Result<Term> {
    let _: Encoding = encoding.try_into()?;

    match binary.decode()? {
        TypedTerm::HeapBinary(heap_binary) => {
            bytes_to_existing_atom(binary, heap_binary.as_bytes())
        }
        TypedTerm::ProcBin(process_binary) => {
            bytes_to_existing_atom(binary, process_binary.as_bytes())
        }
        TypedTerm::BinaryLiteral(binary_literal) => {
            bytes_to_existing_atom(binary, binary_literal.as_bytes())
        }
        TypedTerm::SubBinary(subbinary) => maybe_aligned_maybe_binary_to_atom!(binary, subbinary),
        TypedTerm::MatchContext(match_context) => {
            maybe_aligned_maybe_binary_to_atom!(binary, match_context)
        }
        _ => Err(TypeError)
            .with_context(|| term_is_not_binary("binary", binary))
            .map_err(From::from),
    }
}

fn bytes_to_existing_atom(binary: Term, bytes: &[u8]) -> exception::Result<Term> {
    Atom::try_from_latin1_bytes_existing(bytes)
        .with_context(|| format!("binary ({}) could not be converted to atom", binary))?
        .encode()
        .map_err(From::from)
}
