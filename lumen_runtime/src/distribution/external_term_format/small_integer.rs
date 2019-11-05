use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::{badarg, Process, Term};

use super::{u8, Tag};

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> Result<(Term, &'a [u8]), Exception> {
    let (small_integer_u8, after_small_integer_bytes) = u8::decode(bytes)?;
    let integer = process.integer(small_integer_u8)?;

    Ok((integer, after_small_integer_bytes))
}

pub fn decode_tagged_u8(bytes: &[u8]) -> Result<(u8, &[u8]), Exception> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::SmallInteger => u8::decode(after_tag_bytes),
        _ => Err(badarg!().into()),
    }
}
