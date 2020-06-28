pub mod to_term;

use std::backtrace::Backtrace;
use std::convert::TryInto;
use std::ops::Range;

use anyhow::*;
use thiserror::Error;

use liblumen_alloc::erts::exception::{self, ArcError, Exception, InternalException};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Process;

pub struct PartRange {
    pub byte_offset: usize,
    pub byte_len: usize,
}

impl From<PartRange> for Range<usize> {
    fn from(part_range: PartRange) -> Self {
        part_range.byte_offset..part_range.byte_offset + part_range.byte_len
    }
}

/// Converts `binary` to a list of bytes, each representing the value of one byte.
///
/// ## Arguments
///
/// * `binary` - a heap, reference counted, or subbinary.
/// * `position` - 0-based index into the bytes of `binary`.  `position` can be +1 the last index in
///   the binary if `length` is negative.
/// * `length` - the length of the part.  A negative length will begin the part from the end of the
///   of the binary.
///
/// ## Returns
///
/// * `Ok(Term)` - the list of bytes
/// * `Err(BadArgument)` - binary is not a binary; position is invalid; length is invalid.
pub fn bin_to_list(
    binary: Term,
    position: Term,
    length: Term,
    process: &Process,
) -> exception::Result<Term> {
    let position_usize: usize = position
        .try_into()
        .context("position must be non-negative")?;
    let length_isize: isize = length.try_into().context("length must be an integer")?;

    match binary.decode().unwrap() {
        TypedTerm::HeapBinary(heap_binary) => {
            let available_byte_count = heap_binary.full_byte_len();
            let part_range =
                start_length_to_part_range(position_usize, length_isize, available_byte_count)?;
            let range: Range<usize> = part_range.into();
            let byte_slice: &[u8] = &heap_binary.as_bytes()[range];
            let byte_iter = byte_slice.iter();
            let byte_term_iter = byte_iter.map(|byte| (*byte).into());

            let list = process.list_from_iter(byte_term_iter)?;

            Ok(list)
        }
        TypedTerm::ProcBin(process_binary) => {
            let available_byte_count = process_binary.full_byte_len();
            let part_range =
                start_length_to_part_range(position_usize, length_isize, available_byte_count)?;
            let range: Range<usize> = part_range.into();
            let byte_slice: &[u8] = &process_binary.as_bytes()[range];
            let byte_iter = byte_slice.iter();
            let byte_term_iter = byte_iter.map(|byte| (*byte).into());

            let list = process.list_from_iter(byte_term_iter)?;

            Ok(list)
        }
        TypedTerm::BinaryLiteral(process_binary) => {
            let available_byte_count = process_binary.full_byte_len();
            let part_range =
                start_length_to_part_range(position_usize, length_isize, available_byte_count)?;
            let range: Range<usize> = part_range.into();
            let byte_slice: &[u8] = &process_binary.as_bytes()[range];
            let byte_iter = byte_slice.iter();
            let byte_term_iter = byte_iter.map(|byte| (*byte).into());

            let list = process.list_from_iter(byte_term_iter)?;

            Ok(list)
        }
        TypedTerm::SubBinary(subbinary) => {
            let available_byte_count = subbinary.full_byte_len();
            let part_range =
                start_length_to_part_range(position_usize, length_isize, available_byte_count)?;

            let result = if subbinary.is_aligned() {
                let range: Range<usize> = part_range.into();
                let byte_slice: &[u8] = &unsafe { subbinary.as_bytes_unchecked() }[range];
                let byte_iter = byte_slice.iter();
                let byte_term_iter = byte_iter.map(|byte| (*byte).into());

                process.list_from_iter(byte_term_iter)
            } else {
                let mut byte_iter = subbinary.full_byte_iter();

                // skip byte_offset
                for _ in 0..part_range.byte_offset {
                    byte_iter.next();
                }

                for _ in part_range.byte_len..byte_iter.len() {
                    byte_iter.next_back();
                }

                let byte_term_iter = byte_iter.map(|byte| byte.into());

                process.list_from_iter(byte_term_iter)
            };

            match result {
                Ok(term) => Ok(term),
                Err(error) => Err(error.into()),
            }
        }
        _ => Err(TypeError)
            .context(format!("binary ({}) must be a binary", binary))
            .map_err(From::from),
    }
}

pub fn start_length_to_part_range(
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
