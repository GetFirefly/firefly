// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::string::Encoding;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(binary_to_existing_atom/2)]
pub fn native(binary: Term, encoding: Term) -> exception::Result {
    let _: Encoding = encoding.try_into()?;

    match binary.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => {
            Atom::try_from_latin1_bytes_existing(heap_binary.as_bytes())
                .map_err(|error| error.into())
        }
        TypedTerm::ProcBin(process_binary) => {
            Atom::try_from_latin1_bytes_existing(process_binary.as_bytes())
                .map_err(|error| error.into())
        }
        TypedTerm::SubBinary(subbinary) => {
            if subbinary.is_binary() {
                if subbinary.is_aligned() {
                    let bytes = unsafe { subbinary.as_bytes_unchecked() };

                    Atom::try_from_latin1_bytes_existing(bytes)
                } else {
                    let byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();

                    Atom::try_from_latin1_bytes_existing(&byte_vec)
                }
                .map_err(|error| error.into())
            } else {
                Err(badarg!().into())
            }
        }
        _ => Err(badarg!().into()),
    }
    .map(|atom| unsafe { atom.decode() })
}
