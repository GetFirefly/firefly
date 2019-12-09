use std::backtrace::Backtrace;
use std::str;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;

use super::{atom_utf8, small_atom_utf8, u16, DecodeError, Tag};
use crate::distribution::external_term_format::try_split_at;

pub fn atom_bytes_to_term_bytes((atom, bytes): (Atom, &[u8])) -> (Term, &[u8]) {
    let term: Term = atom.encode().unwrap();

    (term, bytes)
}

pub fn bytes_len_try_into_atom(
    safe: bool,
    bytes: &[u8],
    len: usize,
) -> InternalResult<(Atom, &[u8])> {
    try_split_at(bytes, len).and_then(|(atom_name_bytes, after_atom_name_bytes)| {
        let atom_name = str::from_utf8(atom_name_bytes).context("atom bytes are not UTF-8")?;

        let atom = if safe {
            Atom::try_from_str_existing(atom_name)
        } else {
            Atom::try_from_str(atom_name)
        }
        .context("Could not create atom from bytes")?;

        Ok((atom, after_atom_name_bytes))
    })
}

pub fn bytes_len_try_into_term<'a>(
    safe: bool,
    bytes: &'a [u8],
    len: usize,
) -> InternalResult<(Term, &'a [u8])> {
    bytes_len_try_into_atom(safe, bytes, len).map(atom_bytes_to_term_bytes)
}

pub fn decode_atom<'a>(safe: bool, bytes: &'a [u8]) -> InternalResult<(Atom, &'a [u8])> {
    let (len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = len_u16 as usize;

    bytes_len_try_into_atom(safe, after_len_bytes, len_usize)
}

pub fn decode_tagged<'a>(safe: bool, bytes: &'a [u8]) -> InternalResult<(Atom, &'a [u8])> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::Atom => decode_atom(safe, after_tag_bytes),
        Tag::AtomCacheReference => unimplemented!("{:?}", tag),
        Tag::AtomUTF8 => atom_utf8::decode_atom(safe, after_tag_bytes),
        Tag::SmallAtomUTF8 => small_atom_utf8::decode_atom(safe, after_tag_bytes),
        _ => Err(DecodeError::UnexpectedTag { tag, backtrace: Backtrace::capture() }).context("An atom tag (ATOM_EXT, ATOM_CACHE_REF, ATOM_UTF8_EXT, or SMALL_ATOM_UTF8_EXT) is expected").map_err(|error| error.into()),
    }
}

pub fn decode_term<'a>(safe: bool, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    decode_atom(safe, bytes).map(atom_bytes_to_term_bytes)
}
