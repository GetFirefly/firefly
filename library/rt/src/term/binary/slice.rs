use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ptr;
use core::str;

use crate::term::OpaqueTerm;

use super::*;

/// A slice of another binary or bitstring value
#[repr(C)]
pub struct BitSlice {
    /// This a thin pointer to the original term we're borrowing from
    /// This is necessary to properly keep the owner live, either from the perspective
    /// of the garbage collector, or reference counting, until this slice is no
    /// longer needed.
    ///
    /// If the original data is not from a term, this will be None
    owner: OpaqueTerm,
    /// Fat pointer to a slice of relevant original bytes
    data: *const [u8],
    /// Offset in bits from start of `data`, `0` if this slice starts aligned
    bit_offset: u8,
    /// The number of bits in the underlying data which are relevant to this slice
    num_bits: usize,
}
impl BitSlice {
    pub const TYPE_ID: TypeId = TypeId::of::<BitSlice>();

    #[inline]
    pub fn new(owner: OpaqueTerm, data: &[u8], bit_offset: u8, num_bits: usize) -> Self {
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
            owner,
            data: data as *const [u8],
            bit_offset,
            num_bits,
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
impl Bitstring for BitSlice {
    #[inline]
    fn byte_size(&self) -> usize {
        let total_bits = self.num_bits - self.bit_offset as usize;
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
        &*self.data
    }
}
impl Clone for BitSlice {
    fn clone(&self) -> Self {
        let cloned = Self {
            owner: self.owner,
            data: self.data,
            bit_offset: self.bit_offset,
            num_bits: self.num_bits,
        };

        // If the original owner is reference-counted, we need to increment the strong count
        self.owner.maybe_increment_refcount();

        cloned
    }
}
impl Drop for BitSlice {
    fn drop(&mut self) {
        // If the original owner is reference-counted, we need to decrement the strong count
        self.owner.maybe_decrement_refcount();
    }
}
impl fmt::Debug for BitSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let byte_size = ptr::metadata(self.data);
        f.debug_struct("BitSlice")
            .field("data", &self.data)
            .field("data_size_in_bytes", &byte_size)
            .field("bit_offset", &self.bit_offset)
            .field("bit_size", &self.num_bits)
            .field("is_binary", &self.is_binary())
            .field("is_aligned", &self.is_aligned())
            .finish()
    }
}
impl Eq for BitSlice {}
impl<T: Bitstring> PartialEq<T> for BitSlice {
    fn eq(&self, other: &T) -> bool {
        // An optimization: we can say for sure that if the sizes don't match,
        // the slices don't either.
        if self.bit_size() != other.bit_size() {
            return false;
        }

        // If both slices are aligned binaries, we can compare their data directly
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            let data = unsafe { &*self.data };
            return data.eq(unsafe { other.as_bytes_unchecked() });
        }

        // Otherwise we must fall back to a byte-by-byte comparison
        self.bytes().eq(other.bytes())
    }
}
impl Ord for BitSlice {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<T: Bitstring> PartialOrd<T> for BitSlice {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the standard lib
        if self.is_aligned() && other.is_aligned() && self.is_binary() && other.is_binary() {
            let data = unsafe { &*self.data };
            return Some(data.cmp(unsafe { other.as_bytes_unchecked() }));
        }

        // Otherwise we must comapre byte-by-byte
        Some(self.bytes().cmp(other.bytes()))
    }
}
impl Hash for BitSlice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_aligned() && self.is_binary() {
            return Hash::hash_slice(unsafe { &*self.data }, state);
        }

        for byte in self.bytes() {
            Hash::hash(&byte, state);
        }
    }
}
impl fmt::Display for BitSlice {
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
pub(crate) fn display_bytes<I: Iterator<Item = u8>>(
    mut bytes: I,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    f.write_str("<<")?;

    let Some(byte) = bytes.next() else { return Ok(()); };
    write!(f, "{}", byte)?;

    for byte in bytes {
        write!(f, ",{}", byte)?;
    }

    f.write_str(">>")
}
