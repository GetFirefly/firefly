use liblumen_alloc::gc::GcBox;
use liblumen_alloc::rc::RcBox;

use super::{BinaryFlags, BytesIter, Encoding};

/// This trait provides common behavior for all types which represent
/// binary data, either as a collection of bytes, or a collection of bits.
///
/// Bitstrings are strictly a superset of binaries, as binaries are simply
/// bitstrings which have a number of bits divisible by 8
pub trait Bitstring {
    /// The total number of full bytes required to hold this bitstring
    fn byte_size(&self) -> usize;

    /// The size of the referenced bits in this bitstring.
    ///
    /// This will always be <= `byte_size * 8`, depending on whether
    /// the data is aligned, and whether it is a binary or just a bitstring.
    fn bit_size(&self) -> usize;

    /// The offset of the first demanded bit in the underlying data.
    ///
    /// If not specified, defaults to 0.
    ///
    /// NOTE: The value produced by this function should fall in the range 0..=7
    #[inline]
    fn bit_offset(&self) -> u8 {
        0
    }

    /// This function returns the number of trailing bits in the last byte of
    /// this bitstring, or 0 if there are are none; either because the value
    /// is an aligned binary, or because the final bit aligns on a byte boundary.
    ///
    /// The default is calculated using bit_size + bit_offset % 8
    ///
    /// NOTE: The value produced by this function should fall in the range 0..=7
    fn trailing_bits(&self) -> u8 {
        let total_bits = self.bit_size() + (self.bit_offset() as usize);
        total_bits % 8
    }

    /// Returns an iterator over the bytes in this bitstring.
    ///
    /// See the docs for `BytesIter` for more information on its semantics
    /// around bitstrings.
    fn bytes(&self) -> BytesIter<'_> {
        let data = unsafe { self.as_bytes_unchecked() };
        BytesIter::new(data, self.bit_offset(), self.bit_size())
    }

    /// Returns true if this bitstring begins aligned on a byte boundary.
    ///
    /// Specifically, this means that the first byte of the data is all demanded bits,
    /// so iterating bytes of the bitstring can be performed more efficiently up until
    /// the last byte, which may require masking out discarded bits if the value is not
    /// a binary.
    fn is_aligned(&self) -> bool;

    /// Returns true if this
    fn is_binary(&self) -> bool;

    /// Returns a slice to the memory backing this bitstring directly.
    ///
    /// # Safety
    ///
    /// For bitstrings, accessing the first and last bytes of the returned slice
    /// requires care, as only aligned bitstrings have a fully valid initial byte,
    /// and only aligned binaries are guaranteed to have a fully valid last byte. In
    /// other cases, the first and last bytes may contain bits that should be discarded.
    ///
    /// If you know you have an aligned binary, you should prefer to use `as_bytes`, which
    /// is the safe version of this API. Otherwise, make sure you take appropriate steps
    /// to fix up the first and last bytes depending on alignment/bit_size.
    unsafe fn as_bytes_unchecked(&self) -> &[u8];
}

impl<T: ?Sized + Bitstring, U: Deref<Target = T>> Bitstring for U {
    #[inline]
    fn byte_size(&self) -> usize {
        self.deref().byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.deref().bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        self.deref().trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> BytesIter<'_> {
        self.deref().bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        self.deref().is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.deref().is_binary()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.deref().as_bytes_unchecked()
    }
}

/// This trait provides common behavior for all binary types which represent a collection of bytes
pub trait Binary: Bitstring {
    /// Returns the set of flags that apply to this binary
    fn flags(&self) -> &BinaryFlags;

    /// Returns true if this binary is a raw binary
    #[inline]
    fn is_raw(&self) -> bool {
        self.flags().is_raw()
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    fn is_latin1(&self) -> bool {
        self.flags().is_latin1()
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    fn is_utf8(&self) -> bool {
        self.flags().is_utf8()
    }

    /// Returns a `Encoding` representing the encoding type of this binary
    #[inline]
    fn encoding(&self) -> Encoding {
        self.flags().as_encoding()
    }

    /// Returns the underlying bytes as a raw slice.
    #[inline]
    fn as_bytes(&self) -> &[u8]
    where
        Self: Aligned,
    {
        unsafe { self.as_bytes_unchecked() }
    }

    /// Attempts to access the underlying binary data as a `str`
    ///
    /// Returns `None` if the data is not valid UTF-8.
    ///
    /// If the encoding is known to be UTF-8, this operation is very efficient,
    /// otherwise, the data must be examined for validity first, which is linear
    /// in the size of the string (though validation stops as soon as an error is
    /// encountered).
    #[inline]
    fn as_str(&self) -> Option<&str>
    where
        Self: Aligned,
    {
        if self.flags.is_utf8() {
            Some(unsafe { core::str::from_utf8_unchecked(self.as_bytes()) })
        } else {
            core::str::from_utf8(self.as_bytes()).ok()
        }
    }
}

impl<T: ?Sized + Binary, U: Deref<Target = T>> Binary for U {
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.deref().flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        self.deref().is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.deref().is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        self.deref().is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        self.deref().as_encoding()
    }

    #[inline]
    fn as_bytes(&self) -> &[u8]
    where
        Self: Aligned,
    {
        self.deref().as_bytes()
    }

    #[inline]
    fn as_str(&self) -> Option<&str>
    where
        Self: Aligned,
    {
        self.deref().as_str()
    }
}

/// A marker trait that indicates that the underlying binary/bitstring is
/// always aligned to a byte boundary, enabling safe access to the underlying bytes,
/// and operations such as conversion to `str`.
pub trait Aligned {}

impl<T: ?Sized + Aligned, U: Deref<Target = T>> Aligned for U {}
