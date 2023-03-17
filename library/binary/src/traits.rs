use alloc::borrow::Cow;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Deref;

use half::f16;
use paste::paste;

use crate::helpers::DisplayErlang;
use crate::{BinaryFlags, BitsIter, ByteIter, Encoding, Selection};

/// This trait provides common behavior for all types which represent
/// binary data, either as a collection of bytes, or a collection of bits.
///
/// Bitstrings are strictly a superset of binaries, as binaries are simply
/// bitstrings which have a number of bits divisible by 8
pub trait Bitstring: fmt::Debug {
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
    #[inline]
    fn trailing_bits(&self) -> u8 {
        let total_bits = self.bit_size() + (self.bit_offset() as usize);
        (total_bits % 8) as u8
    }

    /// Obtains a `Selection` containing `n` bytes from this bitstring.
    ///
    /// Returns a `Result<Selection, Selection>` where `Err` implies that there
    /// was insufficient bytes in the underlying data, with a selection covering
    /// what was available.
    #[inline]
    fn select_bytes(&self, n: usize) -> Result<Selection<'_>, Selection<'_>> {
        let data = unsafe { self.as_bytes_unchecked() };
        Selection::new(data, 0, self.bit_offset(), None, n * 8)
    }

    /// Selects `n` bytes from this bitstring, starting at byte `offset`.
    ///
    /// Returns a `Result<Selection, Selection>` where `Err` implies that there
    /// was insufficient bytes in the underlying data, with a selection covering
    /// what was available.
    #[inline]
    fn select_bytes_at(&self, offset: usize, n: usize) -> Result<Selection<'_>, Selection<'_>> {
        let data = unsafe { self.as_bytes_unchecked() };
        Selection::new(data, offset, self.bit_offset(), None, n * 8)
    }

    /// Selects `n` bytes from this bitstring, starting at byte `offset`.
    ///
    /// Negative lengths are supported, and imply a selection seeking backwards from `offset`,
    /// rather than forwards.
    ///
    /// Returns a `Result<Selection, Selection>` where `Err` implies that there
    /// was insufficient bytes in the underlying data, with a selection covering
    /// what was available.
    #[inline]
    fn select_binary_part(&self, offset: usize, n: isize) -> Result<Selection<'_>, Selection<'_>> {
        let data = unsafe { self.as_bytes_unchecked() };
        if n >= 0 {
            let n = n.unsigned_abs();
            Selection::new(data, offset, self.bit_offset(), None, n * 8)
        } else {
            let n = n.unsigned_abs();
            let offset = offset.saturating_sub(n);
            Selection::new(data, offset, self.bit_offset(), None, n * 8)
        }
    }

    /// Returns a tuple of two selections which represent splitting the underlying data in two
    /// at `offset` bytes from the start of the data.
    ///
    /// If the underlying bitstring is empty, both selections will be empty. If `offset` is on
    /// a boundary, such as `0` (the beginning of the data), then the part of the selection on
    /// the boundary will be empty.
    ///
    /// Returns `None` if offset is out of bounds.
    fn select_split(&self, offset: usize) -> Option<(Selection<'_>, Selection<'_>)> {
        let byte_size = self.byte_size();
        match (offset, byte_size) {
            (0, 0) => Some((Selection::Empty, Selection::Empty)),
            (0, _) => Some((Selection::Empty, self.select_all())),
            (n, m) if n == m => Some((self.select_all(), Selection::Empty)),
            (n, _) => {
                let selection = self.select_all();
                let first = selection.take(n * 8).ok()?;
                let second = selection.shrink_front(n * 8);
                Some((first, second))
            }
        }
    }

    /// Obtains a `Selection` containing `n` bits from this bitstring.
    ///
    /// Returns a `Result<Selection, Selection>` where `Err` implies that there
    /// was insufficient bits in the underlying data, with a selection covering
    /// what was available.
    #[inline]
    fn select_bits(&self, n: usize) -> Result<Selection<'_>, Selection<'_>> {
        let data = unsafe { self.as_bytes_unchecked() };
        Selection::new(data, 0, self.bit_offset(), None, n)
    }

    /// Obtains a `Selection` containing all of the bits in the underlying data
    #[inline]
    fn select_all(&self) -> Selection<'_> {
        unsafe { self.select_bits(self.bit_size()).unwrap_unchecked() }
    }

    /// Returns an iterator over the bytes in this bitstring.
    ///
    /// See the docs for `ByteIter` for more information on its semantics
    /// around bitstrings.
    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        ByteIter::new(self.select_all())
    }

    /// Returns an iterator over the full bytes of the underlying bitstring.
    ///
    /// See the docs for `BitsIter` for more information on its semantics
    /// around bitstrings.
    ///
    /// NOTE: The returned iterator requires that the caller handle the possibility of
    /// a trailing partial byte manually, you MUST check the iterator to see if a
    /// trailing byte is present, and handle it accordingly
    #[inline]
    fn bits(&self) -> BitsIter<'_> {
        BitsIter::new(self.select_all())
    }

    /// Returns true if this bitstring begins aligned on a byte boundary.
    ///
    /// This is important when considering how to read the data as bytes. When aligned,
    /// no special treatment is required when reading the data as bytes, except for when
    /// the number of bits is not evenly divisible into bytes. This means that in the optimal
    /// case, we can construct a byte slice very efficiently, and can also cheaply construct
    /// string references (i.e. `str`) from that data when encoded as UTF-8.
    ///
    /// When unaligned, bytes cannot be accessed directly, as each byte requires some number
    /// of bits from adjacent bytes. This precludes creating direct references to the underlying
    /// memory, instead requiring new allocations to transform the data into a more suitable
    /// form.
    #[inline]
    fn is_aligned(&self) -> bool {
        self.bit_offset() == 0
    }

    /// Returns true if this bitstring consists of a number of bits evenly divisible by 8.
    ///
    /// This tells us whether or not there are partial bytes which require special treatment
    /// when reading the data. Binaries require no special treatment when the data is also
    /// aligned; and when unaligned, it is only necessary to account for the offset at which
    /// the byte boundary begins.
    ///
    /// When a bitstring is not a binary, then at least one byte of the underlying data is
    /// partial, meaning some of its bits must be masked out and discarded. When the data is
    /// neither binary nor aligned, both the first and last bytes may be partial bytes, depending
    /// on the offset and number of bits in the data.
    #[inline]
    fn is_binary(&self) -> bool {
        self.bit_size() % 8 == 0
    }

    /// Attempts to access the underlying data as a `str`
    ///
    /// Returns `None` if the bitstring is not aligned and binary, and if the data is not valid
    /// UTF-8
    ///
    /// If the encoding is known to be UTF-8, this operation can be very efficient,
    /// otherwise, the data must be examined for validity first, which is linear
    /// in the size of the string (though validation stops as soon as an error is
    /// encountered).
    #[inline]
    fn as_str(&self) -> Option<&str> {
        if self.is_aligned() && self.is_binary() {
            core::str::from_utf8(unsafe { self.as_bytes_unchecked() }).ok()
        } else {
            None
        }
    }

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

    /// Produces an object which can be used to display a Bitstring using Erlang-style formatting
    fn display(&self) -> DisplayErlang<'_> {
        if self.is_binary() && self.is_aligned() {
            DisplayErlang::Binary(unsafe { self.as_bytes_unchecked() })
        } else {
            DisplayErlang::Iter(self.bytes())
        }
    }
}

impl<B> Bitstring for &B
where
    B: Bitstring + ?Sized,
{
    #[inline]
    fn byte_size(&self) -> usize {
        (**self).byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        (**self).bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        (**self).trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        (**self).bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        (**self).is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        (**self).is_binary()
    }

    #[inline]
    fn as_str(&self) -> Option<&str> {
        (**self).as_str()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        (**self).as_bytes_unchecked()
    }
}

impl<B> Bitstring for Arc<B>
where
    B: Bitstring + ?Sized,
{
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
    fn bytes(&self) -> ByteIter<'_> {
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
    fn as_str(&self) -> Option<&str> {
        self.deref().as_str()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.deref().as_bytes_unchecked()
    }
}

impl<const N: usize> Bitstring for [u8; N] {
    #[inline(always)]
    fn byte_size(&self) -> usize {
        N
    }

    #[inline]
    fn bit_size(&self) -> usize {
        N * 8
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        true
    }

    #[inline]
    fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_slice()).ok()
    }

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Bitstring for [u8] {
    #[inline(always)]
    fn byte_size(&self) -> usize {
        self.len()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.len() * 8
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        true
    }

    #[inline]
    fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self).ok()
    }

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self
    }
}

impl Bitstring for Vec<u8> {
    #[inline(always)]
    fn byte_size(&self) -> usize {
        self.len()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.len() * 8
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        true
    }

    #[inline]
    fn as_str(&self) -> Option<&str> {
        core::str::from_utf8(self.as_slice()).ok()
    }

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Bitstring for str {
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_bytes().len()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_bytes().len() * 8
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        true
    }

    #[inline(always)]
    fn as_str(&self) -> Option<&str> {
        Some(self)
    }

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl Bitstring for String {
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_bytes().len()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_bytes().len() * 8
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        true
    }

    #[inline(always)]
    fn as_str(&self) -> Option<&str> {
        Some(self.as_str())
    }

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// This trait provides common behavior for all types which are always composed of aligned and
/// binary data.
pub trait Binary: Bitstring {
    /// Returns the set of flags that apply to this binary
    fn flags(&self) -> BinaryFlags;

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
    fn as_bytes(&self) -> &[u8] {
        unsafe { self.as_bytes_unchecked() }
    }

    /// Converts the underlying data to UTF-8
    ///
    /// If the data is not UTF-8, it will convert it to valid UTF-8, producing an
    /// owned String containing the converted data. Otherwise, if the data is valid
    /// UTF-8, it simply borrows the underlying data as a str.
    ///
    /// If the encoding is known to be UTF-8, this operation is very efficient,
    /// otherwise, the data must be examined for validity first, which is linear
    /// in the size of the string (though validation stops as soon as an error is
    /// encountered).
    ///
    /// See `Bitstring::as_str` for similar functionality that avoids allocating if the
    /// data is not valid UTF-8.
    #[inline]
    fn to_str(&self) -> Cow<'_, str> {
        if self.is_utf8() {
            Cow::Borrowed(unsafe { core::str::from_utf8_unchecked(self.as_bytes()) })
        } else {
            String::from_utf8_lossy(self.as_bytes())
        }
    }
}

impl<B> Binary for &B
where
    B: Binary + ?Sized,
{
    #[inline]
    fn flags(&self) -> BinaryFlags {
        (**self).flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        (**self).is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        (**self).is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        (**self).is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        (**self).encoding()
    }
}

impl<B> Binary for Arc<B>
where
    B: Binary + ?Sized,
{
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
        self.deref().encoding()
    }
}

impl Binary for [u8] {
    fn flags(&self) -> BinaryFlags {
        let size = self.len();
        let encoding = Encoding::detect(self);
        BinaryFlags::new(size, encoding)
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::detect(self)
    }
}

impl Binary for Vec<u8> {
    fn flags(&self) -> BinaryFlags {
        let size = self.len();
        let encoding = Encoding::detect(self.as_slice());
        BinaryFlags::new(size, encoding)
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::detect(self.as_slice())
    }
}

impl Binary for str {
    #[inline(always)]
    fn flags(&self) -> BinaryFlags {
        let size = self.as_bytes().len();
        BinaryFlags::new(size, Encoding::Utf8)
    }

    #[inline(always)]
    fn encoding(&self) -> Encoding {
        Encoding::Utf8
    }

    #[inline(always)]
    fn is_raw(&self) -> bool {
        false
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.is_ascii()
    }

    #[inline(always)]
    fn is_utf8(&self) -> bool {
        true
    }

    #[inline(always)]
    fn to_str(&self) -> Cow<'_, str> {
        Cow::Borrowed(self)
    }
}
impl Binary for String {
    #[inline(always)]
    fn flags(&self) -> BinaryFlags {
        let size = self.as_bytes().len();
        BinaryFlags::new(size, Encoding::Utf8)
    }

    #[inline(always)]
    fn encoding(&self) -> Encoding {
        Encoding::Utf8
    }

    #[inline(always)]
    fn is_raw(&self) -> bool {
        false
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.is_ascii()
    }

    #[inline(always)]
    fn is_utf8(&self) -> bool {
        true
    }

    #[inline(always)]
    fn to_str(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.as_str())
    }
}

/// A marker trait that indicates that the underlying binary/bitstring is
/// always aligned to a byte boundary, enabling safe access to the underlying bytes,
/// and operations such as conversion to `str`.
pub trait Aligned {}

impl<A: Aligned + ?Sized> Aligned for &A {}
impl<A: Aligned + ?Sized> Aligned for Arc<A> {}

impl<const N: usize> Aligned for [u8; N] {}
impl Aligned for [u8] {}
impl Aligned for str {}
impl Aligned for String {}
impl Aligned for Vec<u8> {}

/// A trait that represents the ability to obtain the byte representation of a endianness-sensitive
/// value, namely numerics.
///
/// This trait can only be implemented on Copy types, and is expected to return an array of bytes of
/// the exact size as the type being converted. This matches how the methods on the standard library
/// numeric types work.
pub trait ToEndianBytes<const N: usize>: Copy {
    /// Converts this value to bytes in big-endian order
    ///
    /// This is a no-op on big-endian systems.
    fn to_be_bytes(self) -> [u8; N];

    /// Converts this value to bytes in little-endian order
    ///
    /// This is a no-op on little-endian systems.
    fn to_le_bytes(self) -> [u8; N];

    /// Converts this value to bytes in native-endian order
    fn to_ne_bytes(self) -> [u8; N];
}

pub trait FromEndianBytes<const N: usize> {
    /// Converts this value to bytes in big-endian order
    ///
    /// This is a no-op on big-endian systems.
    fn from_be_bytes(bytes: [u8; N]) -> Self;

    /// Converts this value to bytes in little-endian order
    ///
    /// This is a no-op on little-endian systems.
    fn from_le_bytes(bytes: [u8; N]) -> Self;

    /// Converts this value to bytes in native-endian order
    fn from_ne_bytes(bytes: [u8; N]) -> Self;
}

macro_rules! primitive_convert_bytes_impl {
    ($($ty:ident),*) => {
        $(
            paste! {
                primitive_convert_bytes_impl_!($ty, [<SIZE_ $ty>]);
            }
        )*
    }
}

macro_rules! primitive_convert_bytes_impl_ {
    ($ty:ident, $sz:ident) => {
        #[allow(non_upper_case_globals)]
        const $sz: usize = core::mem::size_of::<$ty>();

        impl ToEndianBytes<$sz> for $ty {
            #[inline(always)]
            fn to_be_bytes(self) -> [u8; $sz] {
                self.to_be_bytes()
            }

            #[inline(always)]
            fn to_le_bytes(self) -> [u8; $sz] {
                self.to_le_bytes()
            }

            #[inline(always)]
            fn to_ne_bytes(self) -> [u8; $sz] {
                self.to_ne_bytes()
            }
        }

        impl FromEndianBytes<$sz> for $ty {
            #[inline(always)]
            fn from_be_bytes(bytes: [u8; $sz]) -> Self {
                $ty::from_be_bytes(bytes)
            }

            #[inline(always)]
            fn from_le_bytes(bytes: [u8; $sz]) -> Self {
                $ty::from_le_bytes(bytes)
            }

            #[inline(always)]
            fn from_ne_bytes(bytes: [u8; $sz]) -> Self {
                $ty::from_ne_bytes(bytes)
            }
        }
    };
}

primitive_convert_bytes_impl!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f16, f32, f64
);
