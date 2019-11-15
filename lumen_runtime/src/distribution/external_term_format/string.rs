use std::str;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::u16;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = len_u16 as usize;

    if len_usize <= after_len_bytes.len() {
        let (character_bytes, after_characters_bytes) = after_len_bytes.split_at(len_usize);

        match str::from_utf8(character_bytes) {
            Ok(s) => {
                let charlist = process.charlist_from_str(s)?;

                Ok((charlist, after_characters_bytes))
            }
            Err(_) => Err(badarg!().into()),
        }
    } else {
        Err(badarg!().into())
    }
}
