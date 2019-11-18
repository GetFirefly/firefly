use std::str;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;

use super::{atom_utf8, small_atom_utf8, u16, Tag};

pub fn atom_bytes_to_term_bytes((atom, bytes): (Atom, &[u8])) -> (Term, &[u8]) {
    let term: Term = atom.encode().unwrap();

    (term, bytes)
}

pub fn bytes_len_try_into_atom(
    safe: bool,
    bytes: &[u8],
    len: usize,
) -> Result<(Atom, &[u8]), Exception> {
    if len <= bytes.len() {
        let (atom_name_bytes, after_atom_name_bytes) = bytes.split_at(len);

        match str::from_utf8(atom_name_bytes) {
            Ok(atom_name) => {
                let atom = if safe {
                    Atom::try_from_str_existing(atom_name)
                } else {
                    Atom::try_from_str(atom_name)
                }?;

                Ok((atom, after_atom_name_bytes))
            }
            Err(_) => Err(badarg!().into()),
        }
    } else {
        Err(badarg!().into())
    }
}

pub fn bytes_len_try_into_term<'a>(
    safe: bool,
    bytes: &'a [u8],
    len: usize,
) -> Result<(Term, &'a [u8]), Exception> {
    bytes_len_try_into_atom(safe, bytes, len).map(atom_bytes_to_term_bytes)
}

pub fn decode_atom(safe: bool, bytes: &[u8]) -> Result<(Atom, &[u8]), Exception> {
    let (len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = len_u16 as usize;

    bytes_len_try_into_atom(safe, after_len_bytes, len_usize)
}

pub fn decode_tagged<'a>(safe: bool, bytes: &'a [u8]) -> Result<(Atom, &'a [u8]), Exception> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::Atom => decode_atom(safe, after_tag_bytes),
        Tag::AtomCacheReference => unimplemented!("{:?}", tag),
        Tag::AtomUTF8 => atom_utf8::decode_atom(safe, after_tag_bytes),
        Tag::SmallAtomUTF8 => small_atom_utf8::decode_atom(safe, after_tag_bytes),
        _ => Err(badarg!().into()),
    }
}

pub fn decode_term<'a>(safe: bool, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    decode_atom(safe, bytes).map(atom_bytes_to_term_bytes)
}
