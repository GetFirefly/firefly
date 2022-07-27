mod gcbox;

pub use self::gcbox::*;

use core::ptr::Pointee;

use liblumen_binary::{Aligned, Binary, BinaryFlags, Bitstring, ByteIter, Encoding};

impl<T> Bitstring for GcBox<T>
where
    T: ?Sized + Bitstring,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
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

impl<T> Binary for GcBox<T>
where
    T: ?Sized + Binary,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
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

impl<T> Aligned for GcBox<T>
where
    T: ?Sized + Aligned,
    PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
}
