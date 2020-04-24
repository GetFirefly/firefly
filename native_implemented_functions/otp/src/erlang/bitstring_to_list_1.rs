// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

/// Returns a list of integers corresponding to the bytes of `bitstring`. If the number of bits in
/// `bitstring` is not divisible by `8`, the last element of the list is a `bitstring` containing
/// the remaining `1`-`7` bits.
#[native_implemented_function(bitstring_to_list/1)]
pub fn result(process: &Process, bitstring: Term) -> exception::Result<Term> {
    match bitstring.decode()? {
        TypedTerm::HeapBinary(heap_binary) => {
            let byte_term_iter = heap_binary.as_bytes().iter().map(|byte| (*byte).into());
            let last = Term::NIL;

            process
                .improper_list_from_iter(byte_term_iter, last)
                .map_err(|error| error.into())
        }
        TypedTerm::ProcBin(process_binary) => {
            let byte_term_iter = process_binary.as_bytes().iter().map(|byte| (*byte).into());
            let last = Term::NIL;

            process
                .improper_list_from_iter(byte_term_iter, last)
                .map_err(|error| error.into())
        }
        TypedTerm::SubBinary(subbinary) => {
            let last = if subbinary.is_binary() {
                Term::NIL
            } else {
                let partial_byte_subbinary = process.subbinary_from_original(
                    subbinary.original(),
                    subbinary.byte_offset() + subbinary.full_byte_len(),
                    subbinary.bit_offset(),
                    0,
                    subbinary.partial_byte_bit_len(),
                )?;

                process.cons(partial_byte_subbinary, Term::NIL)?
            };

            let byte_term_iter = subbinary.full_byte_iter().map(|byte| byte.into());

            process
                .improper_list_from_iter(byte_term_iter, last)
                .map_err(|error| error.into())
        }
        _ => Err(TypeError)
            .context(format!("bitstring ({}) is not a bitstring", bitstring))
            .map_err(From::from),
    }
}
