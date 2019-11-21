use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;

use super::atom::{atom_bytes_to_term_bytes, bytes_len_try_into_atom};
use super::u16;

pub fn decode_atom(safe: bool, bytes: &[u8]) -> Result<(Atom, &[u8]), Exception> {
    let (len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = len_u16 as usize;

    bytes_len_try_into_atom(safe, after_len_bytes, len_usize)
}

pub fn decode_term(safe: bool, bytes: &[u8]) -> Result<(Term, &[u8]), Exception> {
    decode_atom(safe, bytes).map(atom_bytes_to_term_bytes)
}
