use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::str;

use super::*;

/// Changes the alignment/length of a bit carrier.
#[derive(Debug, Clone)]
pub struct BitSlice<'a> {
    data: &'a [u8],
    /// The number of bits in the underlying data which are relevant to this slice,
    num_bits: usize,
    /// Offset in bits from start of `data`, `0` if this slice starts aligned
    bit_offset: u8,
}
impl<'a> From<&'a [u8]> for BitSlice<'a> {
    #[inline]
    fn from(data: &'a [u8]) -> Self {
        Self::new(data, 0, data.len() * 8)
    }
}
impl<'a> From<&'a str> for BitSlice<'a> {
    #[inline]
    fn from(data: &'a str) -> Self {
        Self::new(data.as_bytes(), 0, data.as_bytes().len() * 8)
    }
}
impl<'a> BitSlice<'a> {
    #[inline]
    pub fn new(data: &'a [u8], bit_offset: u8, num_bits: usize) -> Self {
        assert!(bit_offset < 8, "invalid bit offset, must be a value 0-7");

        let total_bits = num_bits + bit_offset as usize;
        let trailing_bits = total_bits % 8;

        let len = data.len();
        let required_len = (total_bits / 8) + ((trailing_bits > 0) as usize);
        assert!(
            len >= required_len,
            "invalid bit slice range, out of bounds of the given bytes"
        );

        // Shrink the byte slice to only the portion needed
        let data = if len == required_len {
            data
        } else {
            &data[..required_len]
        };

        Self {
            data,
            bit_offset,
            num_bits,
        }
    }

    /// Start a pattern match using this slice
    pub fn matcher(&self) -> Matcher<'a> {
        Matcher::new(self.clone())
    }

    /// Returns the byte at index `n` relative to the start of the data (including bit offset)
    ///
    /// If the index is out of range, returns `None`.
    pub fn get(&self, n: usize) -> Option<u8> {
        if self.bit_offset == 0 {
            if self.num_bits / 8 >= n {
                self.data.get(n).copied()
            } else {
                None
            }
        } else {
            // The byte value we care about is spread across two bytes in the buffer,
            // so we need to mask out the relevant bits from both those bytes and return
            // the reconstructed byte
            if self.num_bits / 8 > n {
                // The shift required to correct for the bit offset
                let left_offset = self.bit_offset;
                // The shift required to move bits from the next byte right, so they can be bitwise-or'd into the output byte
                let right_offset = 8 - left_offset;
                // The mask for bits from the next byte which need to be pulled into the output byte
                let mask = !(-1i8 as u8 >> left_offset);
                let left_bits = self.data[n] << left_offset;
                let right_bits = (self.data[n + 1] & mask) >> right_offset;
                Some(left_bits | right_bits)
            } else {
                // This cannot possibly succeed as there are insufficient bits in the slice
                None
            }
        }
    }

    /// Like `byte`, except panics if the index is out of range.
    pub unsafe fn get_unchecked(&self, n: usize) -> u8 {
        debug_assert!(self.data.len() * 8 > n);

        if self.bit_offset == 0 {
            return *self.data.get_unchecked(n);
        }

        // The shift required to correct for the bit offset
        let left_offset = self.bit_offset;
        // The shift required to move bits from the next byte right, so they can be bitwise-or'd into the output byte
        let right_offset = 8 - left_offset;
        // The mask for bits from the next byte which need to be pulled into the output byte
        let mask = !(-1i8 as u8 >> left_offset);
        let left_bits = self.data.get_unchecked(n) << left_offset;
        let right_bits = (self.data.get_unchecked(n + 1) & mask) >> right_offset;
        left_bits | right_bits
    }

    /// Resizes this slice by shifting the start of the slice forward by `n` bits
    ///
    /// This is used when parsing data from a bitslice to always keep the next value at
    /// the beginning of the slice.
    pub fn advance(&mut self, n: usize) {
        let available = self.num_bits;
        if available < n {
            self.bit_offset = 0;
            self.num_bits = 0;
            self.data = &self.data[..0];
            return;
        }
        // Calculate the change in length (in bytes), this is
        // used to shift the start of the slice forward to its new position
        let mut delta = n / 8;
        // Calculate the new bit length
        let num_bits = available - n;
        // Calculate the new offset, accounting for overflow when the bit
        // offset pushes us into the next byte (i.e. offset > 7)
        let trailing_bits = (num_bits % 8) as u8;
        let mut offset = trailing_bits + self.bit_offset;
        let mut new_len = self.data.len() - delta;
        if offset > 7 {
            new_len -= 1;
            delta += 1;
            offset -= 8;
        }
        self.num_bits = num_bits;
        self.bit_offset = offset;
        std::dbg!(delta);
        unsafe {
            let ptr = self.data.as_ptr().add(delta);
            self.data = core::slice::from_raw_parts(ptr, new_len);
        }
    }

    /// Returns a new slice derived from this slice, with the given number of bits
    pub fn subslice(&self, num_bits: usize) -> Self {
        assert!(self.num_bits >= num_bits);
        let bit_size = (self.bit_offset as usize) + num_bits;
        let len = (bit_size / 8) + (bit_size % 8 > 0) as usize;
        Self {
            data: &self.data[0..=len],
            num_bits,
            bit_offset: self.bit_offset,
        }
    }

    /// Returns the underlying data as a string reference, if the data is binary, aligned, and valid UTF-8.
    ///
    /// For unaligned binaries or bitstrings, returns `None`. See `to_str` for an alternative
    /// available to unaligned binaries.
    pub fn as_str(&self) -> Option<&str> {
        if self.is_aligned() && self.is_binary() {
            str::from_utf8(unsafe { self.as_bytes_unchecked() }).ok()
        } else {
            None
        }
    }

    /// Returns the underlying bytes as a `Cow<str>`, if the data is binary and valid UTF-8.
    ///
    /// The data is borrowed if the underlying bytes can be used directly, otherwise
    /// a new `String` value is constructed.
    ///
    /// For bitstrings, this function returns `None`
    pub fn to_str(&self) -> Option<Cow<'_, str>> {
        if !self.is_binary() {
            return None;
        }
        if self.is_aligned() {
            self.as_str().map(Cow::Borrowed)
        } else {
            let mut buf = Vec::with_capacity(self.num_bits / 8);
            for byte in self.bytes() {
                buf.push(byte);
            }
            String::from_utf8(buf).map(Cow::Owned).ok()
        }
    }
}
impl<'a> Bitstring for BitSlice<'a> {
    #[inline]
    fn byte_size(&self) -> usize {
        let total_bits = self.num_bits + self.bit_offset as usize;
        (total_bits / 8) + ((total_bits % 8 > 0) as usize)
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.num_bits
    }

    #[inline]
    fn bit_offset(&self) -> u8 {
        self.bit_offset
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.data
    }
}

impl<'a> Eq for BitSlice<'a> {}
impl<'a, T: Bitstring> PartialEq<T> for BitSlice<'a> {
    fn eq(&self, other: &T) -> bool {
        // An optimization: we can say for sure that if the sizes don't match,
        // the slices don't either.
        if self.bit_size() != other.bit_size() {
            return false;
        }

        // If both slices are aligned binaries, we can compare their data directly
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            return self.data.eq(unsafe { other.as_bytes_unchecked() });
        }

        // Otherwise we must fall back to a byte-by-byte comparison
        self.bytes().eq(other.bytes())
    }
}
impl<'a> Ord for BitSlice<'a> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<'a, T: Bitstring> PartialOrd<T> for BitSlice<'a> {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the standard lib
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            return Some(self.data.cmp(unsafe { other.as_bytes_unchecked() }));
        }

        // Otherwise we must comapre byte-by-byte
        Some(self.bytes().cmp(other.bytes()))
    }
}
impl<'a> Hash for BitSlice<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_aligned() && self.is_binary() {
            return Hash::hash_slice(self.data, state);
        }

        for byte in self.bytes() {
            Hash::hash(&byte, state);
        }
    }
}
impl<'a> fmt::Display for BitSlice<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        if let Some(s) = self.to_str() {
            f.write_str("<<\"")?;
            for c in s.escape_default() {
                f.write_char(c)?;
            }
            f.write_str("\">>")
        } else {
            display_bytes(self.bytes(), f)
        }
    }
}

/// Displays a raw bitstring using Erlang-style formatting
fn display_bytes<I: Iterator<Item = u8>>(mut bytes: I, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn bitslice_test_get() {
        let bytes = b"hello";
        let slice = BitSlice::new(bytes.as_slice(), 1, 8);
        assert_eq!(slice.get(0), Some(0b11010000));
    }

    #[test]
    fn bitslice_test_advance() {
        let bytes = b"hello";
        let mut slice = BitSlice::new(bytes.as_slice(), 0, 32);
        assert_eq!(slice.get(0), Some(104));
        slice.advance(8);
        assert_eq!(slice.get(0), Some(101));
        slice.advance(8);
        assert_eq!(slice.get(0), Some(108));
        slice.advance(8);
        assert_eq!(slice.get(0), Some(108));
        slice.advance(8);
        assert_eq!(slice.get(0), None);
    }
}
