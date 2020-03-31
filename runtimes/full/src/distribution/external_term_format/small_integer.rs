use std::backtrace::Backtrace;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;

use super::{u8, DecodeError, Tag};

pub fn decode<'a>(process: &Process, bytes: &'a [u8]) -> InternalResult<(Term, &'a [u8])> {
    let (small_integer_u8, after_small_integer_bytes) = u8::decode(bytes)?;
    let integer = process.integer(small_integer_u8)?;

    Ok((integer, after_small_integer_bytes))
}

pub fn decode_tagged_u8<'a>(bytes: &'a [u8]) -> InternalResult<(u8, &'a [u8])> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::SmallInteger => u8::decode(after_tag_bytes),
        _ => Err(DecodeError::UnexpectedTag {
            tag,
            backtrace: Backtrace::capture(),
        })
        .with_context(|| format!("Only {:?} tag expected", Tag::SmallInteger))
        .map_err(|error| error.into()),
    }
}
