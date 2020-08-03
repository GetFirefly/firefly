use std::backtrace::Backtrace;

use liblumen_alloc::erts::exception::InternalResult;

use super::{u8, DecodeError};

pub const NUMBER: u8 = 131;

pub fn check(bytes: &[u8]) -> InternalResult<&[u8]> {
    let (version, after_version_bytes) = u8::decode(bytes)?;

    if version == NUMBER {
        Ok(after_version_bytes)
    } else {
        Err(DecodeError::UnexpectedVersion {
            version,
            backtrace: Backtrace::capture(),
        }
        .into())
    }
}
