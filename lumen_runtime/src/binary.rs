use core::ops::Range;

use std::backtrace::Backtrace;

use thiserror::Error;

use liblumen_alloc::erts::exception::{ArcError, Exception, InternalException};

pub mod to_term;

pub(crate) struct PartRange {
    pub byte_offset: usize,
    pub byte_len: usize,
}

impl From<PartRange> for Range<usize> {
    fn from(part_range: PartRange) -> Self {
        part_range.byte_offset..part_range.byte_offset + part_range.byte_len
    }
}

pub(crate) fn start_length_to_part_range(
    start: usize,
    length: isize,
    available_byte_count: usize,
) -> Result<PartRange, PartRangeError> {
    if start <= available_byte_count {
        if length >= 0 {
            let non_negative_length = length as usize;
            let end = start + non_negative_length;

            if end <= available_byte_count {
                Ok(PartRange {
                    byte_offset: start,
                    byte_len: non_negative_length,
                })
            } else {
                Err(PartRangeError::EndNonNegativeLength {
                    end,
                    available_byte_count,
                    backtrace: Backtrace::capture(),
                })
            }
        } else {
            let start_isize = start as isize;
            let end = start_isize + length;

            if 0 <= start_isize + length {
                let byte_offset = (start_isize + length) as usize;
                let byte_len = (-length) as usize;

                Ok(PartRange {
                    byte_offset,
                    byte_len,
                })
            } else {
                Err(PartRangeError::EndNegativeLength {
                    end,
                    backtrace: Backtrace::capture(),
                })
            }
        }
    } else {
        Err(PartRangeError::Start {
            start,
            available_byte_count,
            backtrace: Backtrace::capture(),
        })
    }
}

#[derive(Debug, Error)]
pub enum PartRangeError {
    #[error("start ({start}) exceeds available_byte_count ({available_byte_count})")]
    Start {
        start: usize,
        available_byte_count: usize,
        backtrace: Backtrace,
    },
    #[error("end ({end}) exceeds available_byte_count ({available_byte_count})")]
    EndNonNegativeLength {
        end: usize,
        available_byte_count: usize,
        backtrace: Backtrace,
    },
    #[error("end ({end}) is less than or equal to 0")]
    EndNegativeLength { end: isize, backtrace: Backtrace },
}

impl From<PartRangeError> for Exception {
    fn from(err: PartRangeError) -> Self {
        InternalException::from(ArcError::from_err(err)).into()
    }
}
