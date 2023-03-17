mod matching;
mod slice;

pub use self::matching::{MatchContext, MatchResult};
pub use self::slice::BitSlice;

use alloc::alloc::{AllocError, Allocator, Global, Layout};
use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use core::any::TypeId;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Index, IndexMut};
use core::ptr::{self, NonNull};
use core::slice::SliceIndex;

use firefly_alloc::heap::Heap;
use firefly_binary::{Aligned, Binary, BinaryFlags, Bitstring, Encoding, Selection};

use crate::gc::Gc;

use super::{Boxable, Header, Metadata, Tag};

/// Empty binary values are used in various places, so for convenience we expose one here
pub const EMPTY_BIN: &'static BinaryData =
    BinaryData::make_constant(BinaryFlags::new(0, Encoding::Raw), &[]);

/// This struct is used to represent both binary _and_ bitstring data.
///
/// Data is always stored aligned, but with a possibly non-zero number of trailing bits.
///
/// The binary flags, contained in the header, can be used to tell whether or not this data
/// is binary or bitstring, and if the latter, how many trailing bits are in the last byte.
#[repr(C)]
pub struct BinaryData {
    header: Header,
    data: [u8],
}
impl<I> Index<I> for BinaryData
where
    I: SliceIndex<[u8]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        index.index(&self.data)
    }
}
impl<I> IndexMut<I> for BinaryData
where
    I: SliceIndex<[u8]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        index.index_mut(&mut self.data)
    }
}
impl BinaryData {
    pub const TYPE_ID: TypeId = TypeId::of::<BinaryData>();

    /// The maximum size of a binary stored on a process heap, in bytes
    pub const MAX_HEAP_BYTES: usize = 64;

    /// Creates a constant utf-8 encoded `BinaryData` value.
    ///
    /// See the usage notes on [`make_constant`]
    pub const fn make_constant_utf8(s: &'static str) -> &'static Self {
        let bytes = s.as_bytes();
        Self::make_constant(BinaryFlags::new(bytes.len(), Encoding::Utf8), bytes)
    }

    /// Creates a constant `BinaryData` value
    ///
    /// This is intended for use at compile-time only, in order to more efficiently construct
    /// strings in the runtime which are used in Erlang code without requiring runtime
    /// allocations
    pub const fn make_constant(flags: BinaryFlags, bytes: &'static [u8]) -> &'static Self {
        use core::intrinsics::const_allocate;

        let size = bytes.len();
        assert!(size == flags.size());
        unsafe {
            let array: *const [u8] = ptr::from_raw_parts(ptr::null(), size);
            let (layout, value_offset) =
                match Layout::new::<Header>().extend(Layout::for_value_raw(array)) {
                    Ok(result) => result,
                    Err(_) => unreachable!(),
                };
            let ptr = const_allocate(layout.size(), layout.align());
            let data_ptr = ptr.add(value_offset);
            let header_ptr = ptr as *mut Header;
            header_ptr.write(Header::new(Tag::Binary, flags.into_raw()));
            ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr, size);
            let ptr: *const BinaryData = ptr::from_raw_parts(ptr.cast_const().cast(), size);
            &*ptr
        }
    }

    /// Overrides the flags/metadata of this binary data
    pub unsafe fn set_flags(&mut self, flags: BinaryFlags) {
        let meta = <BinaryFlags as Metadata<Self>>::pack(flags);
        self.header.set_arity(meta);
    }

    /// Returns the size in bytes of the underlying data
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Copies the bytes from the given slice into `self`
    ///
    /// NOTE: The length of the slice must match the capacity of `self`, or the function will panic
    pub fn copy_from_slice(&mut self, bytes: &[u8]) {
        assert_eq!(self.len(), bytes.len());
        self.data.copy_from_slice(bytes)
    }

    /// Copies the bytes from the given selection into `self`
    ///
    /// NOTE: The length of the selection must match the capacity of `self`, or the function will
    /// panic
    pub fn copy_from_selection(&mut self, selection: Selection<'_>) {
        assert_eq!(self.len(), selection.byte_size());
        let trailing_bits = selection.write_bytes_to_buffer(&mut self.data);
        if trailing_bits > 0 {
            let flags = self.metadata();
            let meta = <BinaryFlags as Metadata<Self>>::pack(
                flags.with_trailing_bits(trailing_bits as usize),
            );
            self.header.set_arity(meta);
        } else {
            let encoding = Encoding::detect(&self.data);
            let meta =
                <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(self.len(), encoding));
            self.header.set_arity(meta);
        }
    }

    /// Constructs a new boxed `BinaryData` from the given string.
    ///
    /// The size of the given string must be <= 64 bytes.
    pub fn from_small_str<A: ?Sized + Allocator>(
        s: &str,
        alloc: &A,
    ) -> Result<Gc<BinaryData>, AllocError> {
        let bytes = s.as_bytes();
        assert!(bytes.len() <= 64);
        let mut boxed = Gc::<BinaryData>::with_capacity_in(bytes.len(), alloc)?;
        {
            let meta = <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(
                bytes.len(),
                Encoding::Utf8,
            ));
            boxed.header = Header::new(Tag::Binary, meta);
            boxed.copy_from_slice(bytes);
        }
        Ok(boxed)
    }

    /// Constructs an Arc<BinaryData> from the given string.
    ///
    /// The encoding of the resulting BinaryData is always UTF-8.
    ///
    /// NOTE: This function always allocates via Arc, even if the binary is smaller than 64 bytes.
    pub fn from_str(s: &str) -> Arc<BinaryData> {
        let bytes = s.as_bytes();
        let byte_size = bytes.len();
        let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), byte_size);
        let layout = unsafe { Layout::for_value_raw(placeholder) };
        let ptr: NonNull<()> = Global.allocate(layout).unwrap().cast();
        let ptr: *mut BinaryData = ptr::from_raw_parts_mut(ptr.as_ptr(), byte_size);
        let mut boxed = unsafe { Box::from_raw(ptr) };
        {
            let meta =
                <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(byte_size, Encoding::Utf8));
            boxed.header = Header::new(Tag::Binary, meta);
            boxed.copy_from_slice(bytes);
        }
        Arc::from(boxed)
    }

    pub fn clone_from_small<A: ?Sized + Allocator>(
        &self,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let mut cloned = Self::with_capacity_small(self.data.len(), alloc)?;
        {
            cloned.header = self.header;
            cloned.data.copy_from_slice(&self.data);
        }
        Ok(cloned)
    }

    pub fn with_capacity_small<A: ?Sized + Allocator>(
        cap: usize,
        alloc: &A,
    ) -> Result<Gc<BinaryData>, AllocError> {
        assert!(cap <= 64);
        let mut gcbox = Gc::<BinaryData>::with_capacity_in(cap, alloc)?;
        {
            let meta = <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(cap, Encoding::Raw));
            gcbox.header = Header::new(Tag::Binary, meta);
        }
        Ok(gcbox)
    }

    pub fn with_capacity_large(cap: usize) -> Arc<BinaryData> {
        assert!(cap > 64);
        let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), cap);
        let layout = unsafe { Layout::for_value_raw(placeholder) };
        let ptr: NonNull<()> = Global.allocate(layout).unwrap().cast();
        let ptr: *mut BinaryData = ptr::from_raw_parts_mut(ptr.as_ptr(), cap);
        let mut boxed = unsafe { Box::from_raw(ptr) };
        {
            let meta = <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(cap, Encoding::Raw));
            boxed.header = Header::new(Tag::Binary, meta);
        }
        Arc::from(boxed)
    }

    /// Constructs a new boxed `BinaryData` from the given bytes.
    ///
    /// The size of the given slice must be <= 64 bytes.
    pub fn from_small_bytes<A: ?Sized + Allocator>(
        bytes: &[u8],
        alloc: &A,
    ) -> Result<Gc<BinaryData>, AllocError> {
        assert!(bytes.len() <= 64);
        let encoding = Encoding::detect(bytes);
        let mut boxed = Gc::<BinaryData>::with_capacity_in(bytes.len(), alloc)?;
        {
            let meta =
                <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(bytes.len(), encoding));
            boxed.header = Header::new(Tag::Binary, meta);
            boxed.copy_from_slice(bytes);
        }
        Ok(boxed)
    }

    /// Constructs an Arc<BinaryData> from the given byte slice.
    ///
    /// The encoding of the given data is detected by examining the bytes. If you
    /// wish to construct a binary from a byte slice with a manually-specified encoding, use
    /// `from_bytes_with_encoding`.
    ///
    /// NOTE: This function always allocates via Arc, even if the binary is smaller than 64 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Arc<BinaryData> {
        let encoding = Encoding::detect(bytes);
        unsafe { Self::from_bytes_with_encoding(bytes, encoding) }
    }

    /// Constructs an Arc<BinaryData> from the given byte slice.
    ///
    /// # Safety
    ///
    /// This function is unsafe because specifying the wrong encoding may violate invariants
    /// of those encodings assumed by other runtime functions. The caller must be sure that
    /// the given bytes are valid for the specified encoding, preferably by having run validation
    /// checks in a previous step.
    pub unsafe fn from_bytes_with_encoding(bytes: &[u8], encoding: Encoding) -> Arc<BinaryData> {
        let byte_size = bytes.len();
        let placeholder: *const BinaryData = ptr::from_raw_parts(ptr::null(), byte_size);
        let layout = unsafe { Layout::for_value_raw(placeholder) };
        let ptr: NonNull<()> = Global.allocate(layout).unwrap().cast();
        let ptr: *mut BinaryData = ptr::from_raw_parts_mut(ptr.as_ptr(), byte_size);
        let mut boxed = unsafe { Box::from_raw(ptr) };
        {
            let meta = <BinaryFlags as Metadata<Self>>::pack(BinaryFlags::new(byte_size, encoding));
            boxed.header = Header::new(Tag::Binary, meta);
            boxed.copy_from_slice(bytes);
        }
        Arc::from(boxed)
    }
}
impl fmt::Debug for BinaryData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}
impl fmt::Display for BinaryData {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        firefly_binary::helpers::display_binary(self, f)
    }
}
impl Eq for BinaryData {}
impl PartialEq for BinaryData {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl crate::cmp::ExactEq for BinaryData {}
impl<T: Bitstring> PartialEq<T> for BinaryData {
    fn eq(&self, other: &T) -> bool {
        // An optimization: we can say for sure that if the sizes don't match,
        // the slices don't either.
        if self.bit_size() != other.bit_size() {
            return false;
        }

        // If both slices are aligned binaries, we can compare their data directly
        if other.is_aligned() && other.is_binary() {
            return self.data.eq(unsafe { other.as_bytes_unchecked() });
        }

        // Otherwise we must fall back to a byte-by-byte comparison
        self.bytes().eq(other.bytes())
    }
}
impl Ord for BinaryData {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.data.cmp(&other.data)
    }
}
impl PartialOrd for BinaryData {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Bitstring> PartialOrd<T> for BinaryData {
    // We order bitstrings lexicographically
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        // Aligned binaries can be compared using the optimal built-in slice comparisons in the
        // standard lib
        if other.is_aligned() && other.is_binary() {
            return Some(self.data.cmp(unsafe { other.as_bytes_unchecked() }));
        }

        // Otherwise we must comapre byte-by-byte
        Some(self.bytes().cmp(other.bytes()))
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
    fn trailing_bits(&self) -> u8 {
        self.metadata().trailing_bits() as u8
    }

    #[inline]
    fn bit_size(&self) -> usize {
        let trailing_bits = self.metadata().trailing_bits();
        (self.len() * 8) - ((8 - trailing_bits) * (trailing_bits > 0) as usize)
    }

    #[inline(always)]
    fn is_aligned(&self) -> bool {
        true
    }

    #[inline(always)]
    fn is_binary(&self) -> bool {
        !self.metadata().is_bitstring()
    }

    #[inline]
    fn as_str(&self) -> Option<&str> {
        if self.is_utf8() {
            Some(unsafe { core::str::from_utf8_unchecked(&self.data) })
        } else {
            core::str::from_utf8(&self.data).ok()
        }
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        &self.data
    }
}
impl Binary for BinaryData {
    #[inline]
    fn flags(&self) -> BinaryFlags {
        self.metadata()
    }

    #[inline]
    fn to_str(&self) -> Cow<'_, str> {
        match self.as_str() {
            Some(s) => Cow::Borrowed(s),
            None => String::from_utf8_lossy(&self.data),
        }
    }
}
impl Aligned for BinaryData {}
impl Boxable for BinaryData {
    type Metadata = BinaryFlags;

    const TAG: Tag = Tag::Binary;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        assert!(self.byte_size() <= Self::MAX_HEAP_BYTES);

        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            return unsafe { Gc::from_raw(ptr.cast_mut()) };
        }

        let mut cloned = Self::with_capacity_small(self.byte_size(), heap).unwrap();
        cloned.header = self.header;
        cloned.data.copy_from_slice(&self.data);
        cloned
    }
}
