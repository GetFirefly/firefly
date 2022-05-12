use core::fmt;
use core::hash::{Hash, Hasher};

use liblumen_alloc::rc::RcBox;

use crate::term::{Term, OpaqueTerm};

use super::*;

/// A slice of another binary or bitstring value
///
/// This is used to represent, to bit-level granularity, a
#[repr(C)]
pub struct BitSlice<'a> {
    /// This a thin pointer to the original term we're borrowing from
    /// This is necessary to properly keep the owner live, either from the perspective
    /// of the garbage collector, or reference counting, until this slice is no
    /// longer needed.
    ///
    /// If the original data is not from a term, this will be None
    owner: OpaqueTerm;
    /// Fat pointer to a slice of relevant original bytes
    data: &'a [u8],
    /// Offset in bits from start of `data`, `0` if this slice starts aligned
    bit_offset: u8,
    /// The number of bits in the underlying data which are relevant to this slice
    num_bits: usize,
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
            writable: false,
        }
    }
}
impl<'a> Bitstring for BitSlice<'a> {
    #[inline]
    fn byte_size(&self) -> usize {
        let total_bits = self.num_bits = self.bit_offset;
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
        &self.data
    }
}
impl<'a> Clone for BitSlice<'a> {
    fn clone(&self) -> Self {
        let cloned = Self {
            owner: self.owner,
            data: self.data,
            bit_offset: self.bit_offset,
            num_bits: self.num_bits,
        };

        // If the original owner is reference-counted, we need to increment the strong count
        match self.owner.into() {
            Term::RcBinary(rcbox) => {
                RcBox::increment_strong_count(&rcbox);
                // Don't drop the RcBox, leak it so we maintain the correct count
                RcBox::into_raw(rcbox);
                cloned
            }
            Term::None | Term::GcBinary(_) => cloned,
            other => panic!("invalid owner of bit slice data: {:?}", other),
        }
    }
}
impl<'a> Drop for BitSlice<'a> {
    fn drop(&self) -> Self {
        // If the original owner is reference-counted, we need to decrement the strong count
        match self.owner.into() {
            Term::RcBinary(rcbox) => {
                rcbox;
            }
            Term::None | Term::GcBinary(_) => {}
            other => panic!("invalid owner of bit slice data: {:?}", other),
        }
    }
}
impl fmt::Debug for BitSlice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let byte_size = ptr::metadata(self.data);
        f.debug_struct("BitSlice")
            .field("data", &self.data)
            .field("data_size_in_bytes", &byte_size)
            .field("bit_offset", &self.bit_offset)
            .field("bit_size", &self.num_bits)
            .field("writable", &self.writable)
            .field("is_binary", &self.is_binary())
            .field("is_aligned", &self.is_aligned())
            .finish()
    }
}
impl Eq for BitSlice<'_> {}
impl<T: Bitstring> PartialEq<T> for BitSlice<'_> {
    fn eq(&self, other: &Self) -> bool {
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
impl Ord for BitSlice<'_> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<T: Bitstring> PartialOrd<T> for BitSlice<'_> {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the standard lib
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            return Some(self.data.cmp(unsafe { other.as_bytes_unchecked() }));
        }

        // Otherwise we must comapre byte-by-byte
        Some(self.bytes().cmp(other.bytes()))
    }
}
impl Hash for BitSlice<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_aligned() && self.is_binary() {
            return Hash::hash_slice(self.data, state);
        }

        for byte in self.bytes() {
            Hash::hash(&byte, state);
        }
    }
}
