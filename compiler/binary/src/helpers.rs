use core::fmt;

use crate::traits::{Aligned, Binary};

/// Creates a mask which can be used to extract `n` bits from a byte,
/// starting from the least-significant bit.
///
/// # Example
///
/// ```rust,ignore
/// let mask = bitmask_le(3);
/// assert_eq!(0b00000111, mask);
/// ```
#[inline(always)]
pub const fn bitmask_le(n: u8) -> u8 {
    debug_assert!(n <= 8);
    u8::MAX.checked_shr(8 - (n as u32)).unwrap_or(0)
}

/// Creates a mask which can be used to extract `n` bits from a byte,
/// starting from the most-significant bit.
///
/// # Example
///
/// ```rust,ignore
/// let mask = bitmask_be(3);
/// assert_eq!(0b11100000, mask);
/// ```
#[inline(always)]
pub const fn bitmask_be(n: u8) -> u8 {
    debug_assert!(n <= 8);
    u8::MAX.checked_shl(8 - (n as u32)).unwrap_or(0)
}

/// Combine bits from bytes `x` and `y` at the given offset.
///
/// The resulting byte will contain `8 - offset` bits from `x` starting
/// from the offset bit, and taking any remaining bits from `y` starting
/// with the most-significant bit.
///
/// # Example
///
/// ```rust,ignore
/// let offset = 6;
/// let x = 0b10000111;
/// let y = 0b01000010;
/// let z = splice_bits(x, y, offset);
/// assert_eq!(0b110100000, z);
/// ```
#[inline(always)]
pub fn splice_bits(x: u8, y: u8, offset: u8) -> u8 {
    debug_assert!(offset <= 8);
    let inverse_offset = 8 - offset;
    let offset_mask = bitmask_le(inverse_offset);
    let offset = offset as u32;
    let left_bits = (x & offset_mask).checked_shl(offset).unwrap_or(0);
    let right_bits = (y & !offset_mask)
        .checked_shr(inverse_offset as u32)
        .unwrap_or(0);
    left_bits | right_bits
}

/// Calculates a new index and offset in a byte slice, given the current index
/// and offset, and the number of bits that were consumed.
///
/// The number of bits consumed may span multiple bytes, and it is assumed that
/// consumption started at bit `(index * 8) + bit_offset` of the underlying memory.
///
/// The offset must be <= 8, or this function will panic
///
/// This function will also panic if the index would overflow, but as that would
/// imply an impermissably large bitstring, a panic in this situation is guaranteed
/// anyway.
///
/// # Example
///
/// ```rust,ignore
/// let index = 0;
/// let offset = 1;
/// let bits_consumed = 5
/// assert_eq!(next_index(index, offset, bits_consumed), (0, 6));
/// assert_eq!(next_index(0, 3, bits_consumed), (1, 0));
/// ```
#[inline]
#[allow(unused)]
pub fn next_index(index: usize, bit_offset: u8, bits_consumed: usize) -> (usize, u8) {
    // Calculate the number of bytes consumed, this is the base delta
    let delta_bytes = bits_consumed / 8;
    let next_index = index + delta_bytes;
    // Calculate the number of trailing bits consumed.
    let delta_bits = (bits_consumed % 8) as u8;
    // When added to the current offset, we determine whether those
    // trailing bits would have traversed an extra byte boundary.
    let next_offset = bit_offset + delta_bits;
    if next_offset > 7 {
        // The trailing bits overflowed into the next byte, so our
        // index delta increases by 1, and our new offset is determined
        // by the subtracting the excess bits from the combined offset
        (next_index + 1, next_offset - 8)
    } else {
        // The trailing bits stayed within the same byte, so we're done!
        (next_index, next_offset)
    }
}

/// This struct is used to provide a common renderer for Erlang bitstrings
pub enum DisplayErlang<'a> {
    Binary(&'a [u8]),
    Iter(crate::ByteIter<'a>),
}

impl<'a> fmt::Display for DisplayErlang<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Binary(bytes) => match core::str::from_utf8(bytes) {
                Ok(s) => display_binary(s, f),
                Err(_) => display_bytes(bytes.iter().copied(), f),
            },
            Self::Iter(iter) => display_bytes(iter.clone(), f),
        }
    }
}

/// Displays an aligned Binary using Erlang-style formatting
pub fn display_binary<B: Binary + Aligned>(bin: B, f: &mut fmt::Formatter) -> fmt::Result {
    use core::fmt::Write;

    if let Some(s) = bin.as_str() {
        f.write_str("<<\"")?;
        for c in s.escape_default() {
            f.write_char(c)?;
        }
        f.write_str("\">>")
    } else {
        display_bytes(bin.as_bytes().iter().copied(), f)
    }
}

/// Displays a sequence of raw bytes using Erlang-style formatting
pub fn display_bytes<I: Iterator<Item = u8>>(mut bytes: I, f: &mut fmt::Formatter) -> fmt::Result {
    f.write_str("<<")?;

    let Some(byte) = bytes.next() else { return Ok(()); };
    write!(f, "{}", byte)?;

    for byte in bytes {
        write!(f, ",{}", byte)?;
    }

    f.write_str(">>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_test_bitmask_le() {
        assert_eq!(bitmask_le(0), 0b00000000);
        assert_eq!(bitmask_le(1), 0b00000001);
        assert_eq!(bitmask_le(2), 0b00000011);
        assert_eq!(bitmask_le(3), 0b00000111);
        assert_eq!(bitmask_le(4), 0b00001111);
        assert_eq!(bitmask_le(5), 0b00011111);
        assert_eq!(bitmask_le(6), 0b00111111);
        assert_eq!(bitmask_le(7), 0b01111111);
        assert_eq!(bitmask_le(8), 0b11111111);
    }

    #[test]
    fn helper_test_bitmask_be() {
        assert_eq!(bitmask_be(0), 0b00000000);
        assert_eq!(bitmask_be(1), 0b10000000);
        assert_eq!(bitmask_be(2), 0b11000000);
        assert_eq!(bitmask_be(3), 0b11100000);
        assert_eq!(bitmask_be(4), 0b11110000);
        assert_eq!(bitmask_be(5), 0b11111000);
        assert_eq!(bitmask_be(6), 0b11111100);
        assert_eq!(bitmask_be(7), 0b11111110);
        assert_eq!(bitmask_be(8), 0b11111111);
    }

    #[test]
    fn helper_test_splice_bits() {
        const X: u8 = 0b10010011;
        const Y: u8 = 0b01101101;
        assert_eq!(splice_bits(X, Y, 0), X);
        assert_eq!(splice_bits(X, Y, 1), 0b00100110);
        assert_eq!(splice_bits(X, Y, 2), 0b01001101);
        assert_eq!(splice_bits(X, Y, 3), 0b10011011);
        assert_eq!(splice_bits(X, Y, 4), 0b00110110);
        assert_eq!(splice_bits(X, Y, 5), 0b01101101);
        assert_eq!(splice_bits(X, Y, 6), 0b11011011);
        assert_eq!(splice_bits(X, Y, 7), 0b10110110);
        assert_eq!(splice_bits(X, Y, 8), Y);
    }

    #[test]
    fn helper_test_next_index() {
        let mut index = 0;
        let mut offset = 0;
        let mut consumed = 0;
        assert_eq!(next_index(index, offset, consumed), (0, 0));
        consumed += 1;
        assert_eq!(next_index(index, offset, consumed), (0, 1));
        offset += 1;
        assert_eq!(next_index(index, offset, consumed), (0, 2));
        consumed += 5;
        assert_eq!(next_index(index, offset, consumed), (0, 7));
        consumed += 1;
        assert_eq!(next_index(index, offset, consumed), (1, 0));
        offset += 6;
        assert_eq!(next_index(index, offset, consumed), (1, 6));
        consumed += 1;
        assert_eq!(next_index(index, offset, consumed), (1, 7));
        index += 1;
        assert_eq!(next_index(index, offset, consumed), (2, 7));
    }
}
