use core::ptr::Pointee;

use liblumen_alloc::gc::{self, GcBox};
use liblumen_alloc::rc::{self, Rc, Weak};

use super::{BinaryFlags, ByteIter, Encoding};

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
    #[inline]
    fn trailing_bits(&self) -> u8 {
        let total_bits = self.bit_size() + (self.bit_offset() as usize);
        (total_bits % 8) as u8
    }

    /// Returns an iterator over the bytes in this bitstring.
    ///
    /// See the docs for `ByteIter` for more information on its semantics
    /// around bitstrings.
    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        let data = unsafe { self.as_bytes_unchecked() };
        ByteIter::new(data, self.bit_offset(), self.bit_size())
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
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        (**self).as_bytes_unchecked()
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

    #[inline(always)]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self
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
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<T> Bitstring for GcBox<T>
where
    T: ?Sized + Bitstring,
    gc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_ref().byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_ref().bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        self.as_ref().trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        self.as_ref().bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        self.as_ref().is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.as_ref().is_binary()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_ref().as_bytes_unchecked()
    }
}

impl<T> Bitstring for Rc<T>
where
    T: ?Sized + Bitstring,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_ref().byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_ref().bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        self.as_ref().trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        self.as_ref().bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        self.as_ref().is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.as_ref().is_binary()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_ref().as_bytes_unchecked()
    }
}
impl<T> Bitstring for Weak<T>
where
    T: ?Sized + Bitstring,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn byte_size(&self) -> usize {
        self.as_ref().byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.as_ref().bit_size()
    }

    #[inline]
    fn trailing_bits(&self) -> u8 {
        self.as_ref().trailing_bits()
    }

    #[inline]
    fn bytes(&self) -> ByteIter<'_> {
        self.as_ref().bytes()
    }

    #[inline]
    fn is_aligned(&self) -> bool {
        self.as_ref().is_aligned()
    }

    #[inline]
    fn is_binary(&self) -> bool {
        self.as_ref().is_binary()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.as_ref().as_bytes_unchecked()
    }
}

/// This trait provides common behavior for all binary types which represent a collection of bytes
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
        if self.is_utf8() {
            Some(unsafe { core::str::from_utf8_unchecked(self.as_bytes()) })
        } else {
            core::str::from_utf8(self.as_bytes()).ok()
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
    fn as_str(&self) -> Option<&str>
    where
        Self: Aligned,
    {
        Some(self)
    }
}

impl<T> Binary for GcBox<T>
where
    T: ?Sized + Binary,
    gc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.as_ref().flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        self.as_ref().is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.as_ref().is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        self.as_ref().is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        self.as_ref().encoding()
    }
}

impl<T> Binary for Rc<T>
where
    T: ?Sized + Binary,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.as_ref().flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        self.as_ref().is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.as_ref().is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        self.as_ref().is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        self.as_ref().encoding()
    }
}

impl<T> Binary for Weak<T>
where
    T: ?Sized + Binary,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.as_ref().flags()
    }

    #[inline]
    fn is_raw(&self) -> bool {
        self.as_ref().is_raw()
    }

    #[inline]
    fn is_latin1(&self) -> bool {
        self.as_ref().is_latin1()
    }

    #[inline]
    fn is_utf8(&self) -> bool {
        self.as_ref().is_utf8()
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        self.as_ref().encoding()
    }
}

/// A marker trait that indicates that the underlying binary/bitstring is
/// always aligned to a byte boundary, enabling safe access to the underlying bytes,
/// and operations such as conversion to `str`.
pub trait Aligned {}

impl<A: Aligned + ?Sized> Aligned for &A {}

impl Aligned for [u8] {}
impl Aligned for str {}

impl<T> Aligned for GcBox<T>
where
    T: ?Sized + Aligned,
    gc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}
impl<T> Aligned for Rc<T>
where
    T: ?Sized + Aligned,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}
impl<T> Aligned for Weak<T>
where
    T: ?Sized + Aligned,
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}
