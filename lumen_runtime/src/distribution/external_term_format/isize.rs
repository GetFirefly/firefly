use std::backtrace::Backtrace;

use anyhow::*;

use liblumen_alloc::erts::exception::InternalResult;

use super::{i32, u8, DecodeError, Tag};

pub fn decode(bytes: &[u8]) -> InternalResult<(isize, &[u8])> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::Integer => {
            let (i, after_i32_bytes) = i32::decode(after_tag_bytes)?;

            Ok((i as isize, after_i32_bytes))
        }
        Tag::SmallInteger => {
            let (u, after_u8_bytes) = u8::decode(after_tag_bytes)?;

            Ok((u as isize, after_u8_bytes))
        }
        _ => Err(DecodeError::UnexpectedTag {
            tag,
            backtrace: Backtrace::capture(),
        })
        .with_context(|| format!("Expected {} or {} tag", Tag::Integer, Tag::SmallInteger))
        .map_err(|error| error.into()),
    }
}
