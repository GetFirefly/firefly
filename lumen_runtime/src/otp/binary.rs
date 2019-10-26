use core::convert::TryInto;
use core::ops::Range;

use liblumen_alloc::erts::exception::Result;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{badarg, Process};

use crate::binary::start_length_to_part_range;

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
pub fn bin_to_list(binary: Term, position: Term, length: Term, process: &Process) -> Result {
    let position_usize: usize = position.try_into()?;
    let length_isize: isize = length.try_into()?;

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
        _ => Err(badarg!().into()),
    }
}
