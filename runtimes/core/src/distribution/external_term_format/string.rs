use std::str;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::u16;
use crate::distribution::external_term_format::try_split_at;

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (len_u16, after_len_bytes) = u16::decode(bytes)?;
    let len_usize = len_u16 as usize;

    try_split_at(after_len_bytes, len_usize).and_then(
        |(character_bytes, after_characters_bytes)| {
            let s = str::from_utf8(character_bytes).context("string is not UTF-8")?;
            let charlist = process.charlist_from_str(s)?;

            Ok((charlist, after_characters_bytes))
        },
    )
}
