mod flags;
mod iter;
mod slice;
mod traits;
mod writer;

pub use self::flags::{BinaryFlags, Encoding};
pub use self::iter::BytesIter;
pub use self::slice::BitSlice;
pub use self::traits::{Aligned, Binary, Bitstring};
pub use self::writer::BinaryWriter;

use core::any::TypeId;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Index, IndexMut};
use core::slice::SliceIndex;
use core::sync::atomic::AtomicUsize;

use liblumen_alloc::gc::GcBox;
use liblumen_alloc::rc::RcBox;

/// A Gc binary is one that is 64 bytes or less and is allocated directly
/// on the heap of the owning process, and is cleaned up by the garbage
/// collector directly.
pub type GcBinary = GcBox<BinaryData>;

/// An Rc binary is any binary larger than 64 bytes, is allocated on
/// the global heap, and reference-counted, only being cleaned up when
/// the last reference is dropped.
pub type RcBinary = RcBox<BinaryData>;

/// This represents binary data, i.e. byte-aligned, with a number of bits
/// divisible by 8 evenly.
pub struct BinaryData {
    flags: BinaryFlags,
    data: [u8],
}
impl<I> Index<I> for BinaryData where I: SliceIndex<[u8]> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        index.index(&self.data)
    }
}
impl<I> IndexMut<I> for BinaryData where SliceIndex<[u8]> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(&mut self.data)
    }
}
impl BinaryData {
    pub const TYPE_ID: TypeId = TypeId::of::<Binary>();

    /// Overrides the flags/metadata of this binary data
    pub unsafe fn set_flags(&mut self, flags: BinaryFlags) {
        self.flags = flags;
    }

    /// Returns the size in bytes of the underlying data
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a BinaryWriter which can write to this binary.
    #[inline]
    pub fn write<'a>(&'a mut self) -> BinaryWriter<'a> {
        BinaryWriter::new(&mut self.data)
    }

    /// Copies the bytes from the given slice into `self`
    ///
    /// NOTE: The length of the slice must match the capacity of `self`, or the function will panic
    pub fn copy_from_slice(&mut self, bytes: &[u8]) {
        assert_eq!(self.len(), bytes.len());
        self.data.copy_from_slice(bytes)
    }

    /// Constructs an RcBinary from the given string.
    ///
    /// The encoding of the resulting BinaryData is always UTF-8.
    ///
    /// NOTE: This function always allocates via RcBox, even if the binary is smaller than 64 bytes.
    pub fn from_str(s: &str) -> RcBinary {
        let bytes = s.as_bytes();
        let mut rcbox = RcBox::<BinaryData>::with_capacity(bytes.len());
        {
            let value = unsafe { RcBox::get_mut_unchecked(&mut rcbox) };
            value.flags = BinaryFlags::new(Encoding::Utf8);
            value.copy_from_slice(bytes);
        }
        rcbox
    }

    /// Constructs an RcBinary from the given byte slice.
    ///
    /// The encoding of the given data is detected by examining the bytes. If you
    /// wish to construct a binary from a byte slice with a manually-specified encoding, use
    /// `from_bytes_with_encoding`.
    ///
    /// NOTE: This function always allocates via RcBox, even if the binary is smaller than 64 bytes.
    pub fn from_bytes(bytes: &[u8]) -> RcBinary {
        let encoding = Encoding::detect(bytes);
        unsafe { from_bytes_with_encoding(bytes, encoding) }
    }

    /// Constructs an RcBinary from the given byte slice.
    ///
    /// # Safety
    ///
    /// This function is unsafe because specifying the wrong encoding may violate invariants
    /// of those encodings assumed by other runtime functions. The caller must be sure that
    /// the given bytes are valid for the specified encoding, preferably by having run validation
    /// checks in a previous step.
    pub unsafe fn from_bytes_with_encoding(bytes: &[u8], encoding: Encoding) -> RcBinary {
        let mut rcbox = RcBox::<BinaryData>::with_capacity(bytes.len());
        {
            let value = unsafe { RcBox::get_mut_unchecked(&mut rcbox) };
            value.flags = BinaryFlags::new(encoding);
            value.copy_from_slice(bytes);
        }
        rcbox
    }

    #[inline]
    pub fn full_byte_iter<'a>(&'a self) -> iter::Copied<slice::Iter<'a, u8>> {
        self.data.iter().copied()
    }
}
impl fmt::Debug for BinaryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}
impl fmt::Display for BinaryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(s) = self.as_str() {
            write!(f, "<<\"{}\">>", s.escape_default().to_string())
        } else {
            display_bytes(self.as_bytes(), f)
        }
    }
}
impl Eq for BinaryData {}
impl PartialEq for BinaryData {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl PartialOrd for BinaryData {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BinaryData {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}

impl Hash for BinaryData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash_slice(&self.data, state);
    }
}
impl Bitstring for BinaryData {
    #[inline]
    fn byte_size(&self) -> usize {
        self.len()
    }

    #[inline]
    unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
}
impl Binary for BinaryData {
    #[inline]
    fn flags(&self) -> &BinaryFlags {
        &self.flags
    }
}
impl AlignedBinary for BinaryData {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}
impl MaybePartialByte for BinaryData {
    #[inline]
    fn partial_byte_bit_len(&self) -> u8 {
        0
    }

    #[inline]
    fn total_bit_len(&self) -> usize {
        self.len() * 8
    }

    #[inline]
    fn total_byte_len(&self) -> usize {
        self.len()
    }
}
impl IndexByte for BinaryData {
    #[inline]
    fn byte(&self, index: usize) -> u8 {
        self.data[index]
    }
}

/// Displays a raw binary using Erlang-style formatting
pub(crate) fn display_bytes(
    bytes: &[u8],
    encoding: Encoding,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    f.write_str("<<")?;

    let mut iter = bytes.iter().copied();

    let Some(byte) = iter.next() else { return Ok(()); };
    write!(f, "{}", byte)?;

    for byte in iter {
        write!(f, ",{}", byte)?;
    }

    f.write_str(">>")
}
