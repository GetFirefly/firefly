pub mod to_term;

use core::ops::Range;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;

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
    process: &Process,
    start: usize,
    length: isize,
    available_byte_count: usize,
) -> exception::Result<PartRange> {
    if length >= 0 {
        let non_negative_length = length as usize;

        if (start <= available_byte_count) && (start + non_negative_length <= available_byte_count)
        {
            Ok(PartRange {
                byte_offset: start,
                byte_len: non_negative_length,
            })
        } else {
            Err(badarg!(process).into())
        }
    } else {
        let start_isize = start as isize;

        if (start <= available_byte_count) && (0 <= start_isize + length) {
            let byte_offset = (start_isize + length) as usize;
            let byte_len = (-length) as usize;

            Ok(PartRange {
                byte_offset,
                byte_len,
            })
        } else {
            Err(badarg!(process).into())
        }
    }
}
