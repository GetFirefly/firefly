use alloc::alloc::{AllocError, Layout};
use core::fmt;
use core::ptr::Pointee;

use firefly_alloc::heap::Heap;
use firefly_binary::{BinaryFlags, Bitstring};

use crate::gc::Gc;

use super::OpaqueTerm;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Tag {
    BigInt = 0b0000,
    Tuple = 0b0001,
    Map = 0b0010,
    Closure = 0b0011,
    Pid = 0b0100,
    Port = 0b0101,
    Reference = 0b0110,
    Binary = 0b0111,
    Slice = 0b1000,
    Match = 0b1001,
}

/// This trait is implemented by types which can be encoded
/// in the arity bits of a `Header` value. The only requirements on
/// implementations of this trait are the following:
///
/// * Must be able to be encoded in 45 bits
/// * Must be able to reify metadata that can be used with `core::ptr::from_raw_parts`
/// to produce a valid pointer to the containing type.
pub trait Metadata<T: ?Sized>: Copy {
    /// Returns the pointee metadata for the containing type.
    ///
    /// In many cases this corresponds to the arity of the underlying type, if it is unsized,
    /// but that is not always the case. In all cases however, this value can be used with
    /// `core::ptr::from_raw_parts` to reconstitute a valid pointer to the pointee.
    fn metadata(&self) -> <T as Pointee>::Metadata;
    /// Encodes this metadata value as a `usize`
    fn pack(self) -> usize;
    /// Decodes this metadata from a `usize`
    unsafe fn unpack(raw: usize) -> Self;
}
/// Implements `Metadata` for any unsized, slice-like type with a `usize` representing its capacity
impl<T> Metadata<T> for usize
where
    T: ?Sized + Pointee<Metadata = usize>,
{
    #[inline(always)]
    fn metadata(&self) -> <T as Pointee>::Metadata {
        *self
    }
    #[inline(always)]
    fn pack(self) -> usize {
        self
    }
    #[inline(always)]
    unsafe fn unpack(raw: usize) -> Self {
        raw
    }
}
/// Implements `Metadata` for any sized type using a placeholder `usize` value
impl<T> Metadata<T> for ()
where
    T: Pointee<Metadata = ()>,
{
    #[inline(always)]
    fn metadata(&self) -> <T as Pointee>::Metadata {}
    #[inline(always)]
    fn pack(self) -> usize {
        0
    }
    #[inline(always)]
    unsafe fn unpack(_: usize) -> Self {
        ()
    }
}
impl<T> Metadata<T> for BinaryFlags
where
    T: ?Sized + Pointee<Metadata = usize> + Bitstring,
{
    #[inline(always)]
    fn metadata(&self) -> <T as Pointee>::Metadata {
        self.size()
    }
    #[inline(always)]
    fn pack(self) -> usize {
        self.into_raw()
    }
    #[inline(always)]
    unsafe fn unpack(raw: usize) -> Self {
        Self::from_raw(raw)
    }
}

pub trait Boxable {
    /// Represents the type of value encoded in the arity bits of a `Header`
    type Metadata: Metadata<Self>;

    const TAG: Tag;

    fn header(&self) -> &Header;

    fn header_mut(&mut self) -> &mut Header;

    /// Returns true if the header term has been rewritten as a forwarding pointer
    #[inline]
    fn is_moved(&self) -> bool {
        self.header().is_moved()
    }

    /// Returns the forwarding pointer if the term has been moved
    ///
    /// This is unsafe to call if you haven't checked `is_moved` yet
    #[inline]
    fn forwarded_to(&self) -> *mut () {
        self.header().forwarded_to()
    }

    /// Returns the metadata for the current value
    #[inline]
    fn metadata(&self) -> Self::Metadata {
        unsafe { <Self as Boxable>::Metadata::unpack(self.header().arity()) }
    }

    /// This is equivalent to `core::ptr::from_raw_parts`, but instead of specifying
    /// the Rust metadata for the pointer, the caller can provide a `Header` and let
    /// the implementation of this trait decode that header into the Rust metadata as
    /// needed.
    unsafe fn from_raw_parts<'a>(ptr: *mut (), header: Header) -> *mut Self {
        assert_eq!(header.tag(), Self::TAG);
        let metadata = <Self as Boxable>::Metadata::unpack(header.arity());
        core::ptr::from_raw_parts_mut(ptr, metadata.metadata())
    }

    /// Calculates an appropriate [`Layout`] which can contain this type as well
    /// as any data it references which is also heap-allocated (i.e. not reference counted)
    ///
    /// This function assumes that all the data must be cloned, but see [`layout_excluding_heap`]
    /// for a version which takes into consideration data which is already allocated on a
    /// given heap.
    ///
    /// By default this is implemented with [`Layout::for_value`], but that does not take into account
    /// any data contained by this type. This is still a sane default for most types.
    #[inline]
    fn layout(&self) -> Layout {
        use firefly_alloc::heap::EmptyHeap;

        self.layout_excluding_heap(&EmptyHeap)
    }
    /// Calculates an appropriate [`Layout`] which can contain this type as well
    /// as any data it references which is also heap-allocated (i.e. not reference counted), _and_
    /// is not already allocated on `heap`.
    ///
    /// This is useful for situations in which you need to clone some data to a heap which may
    /// already be on that heap, or which may reference data already on that heap, avoiding redundant
    /// allocations.
    ///
    /// By default this is implemented by checking if the `self` reference is an address within the
    /// range owned by `heap`, and returning an empty [`Layout`] if so. Otherwise it delegates to
    /// [`layout`].
    ///
    /// NOTE: The resulting [`Layout`] must be used immediately to clone `self` to the heap, as
    /// it is reliant on the current state of the heap, which may change at any time.
    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains(self as *const Self as *const ()) {
            Layout::new::<()>()
        } else {
            Layout::for_value(self)
        }
    }

    /// This function can be used to clone `self` to `heap`.
    ///
    /// Implementations must check that there is sufficient space on the target heap to hold `self`
    /// and all of its transitive heap-allocated children _before_ starting to clone. If there is
    /// not enough space, the function should return `Err`.
    ///
    /// If an attempted allocation fails during the course of this function, the function MUST panic.
    ///
    /// Returns a pointer to the cloned item if successful, otherwise returns an allocation error.
    ///
    /// # SAFETY
    ///
    /// Callers must be able to safely call this function without checking if there is available
    /// heap to fulfill the request. As such, implementations must determine the total size of the
    /// data to be cloned and verify the target heap has that much space available. The default
    /// implementation does this automatically for you.
    fn clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Result<Gc<Self>, AllocError> {
        let layout = self.layout_excluding_heap(heap);
        if heap.heap_available() < layout.size() {
            return Err(AllocError);
        }
        Ok(self.unsafe_clone_to_heap(heap))
    }

    /// This function is an unsafe form of [`clone_to_heap`], in that it makes no effort to
    /// check if there is available space on the target heap before cloning into it. If the
    /// clone would fail due to an allocation error, implementations MUST panic.
    ///
    /// # SAFETY
    ///
    /// This function assumes that the caller has already computed a [`Layout`] for `self` on `heap`,
    /// and that `heap` has sufficient space available to contain it. Implementations are free
    /// to assume that this is the case.
    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self>;
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Header(OpaqueTerm);
impl fmt::Debug for Header {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Header")
            .field("tag", &self.tag())
            .field("arity", &self.arity())
            .finish()
    }
}
impl Header {
    #[inline]
    pub const fn new(tag: Tag, arity: usize) -> Self {
        Self(OpaqueTerm::header(tag, arity))
    }

    #[inline]
    pub fn tag(&self) -> Tag {
        unsafe { self.0.tag() }
    }

    /// Returns true if the header term has been rewritten as a forwarding pointer
    #[inline]
    pub const fn is_moved(&self) -> bool {
        self.0.is_box()
    }

    #[inline]
    pub const fn forwarded_to(&self) -> *mut () {
        assert!(self.is_moved());
        unsafe { self.0.as_ptr() }
    }

    #[inline(always)]
    pub const fn arity(&self) -> usize {
        unsafe { self.0.arity() }
    }

    pub fn set_arity(&mut self, arity: usize) {
        self.0 = OpaqueTerm::header(self.tag(), arity);
    }
}
impl From<OpaqueTerm> for Header {
    fn from(raw: OpaqueTerm) -> Self {
        assert!(raw.is_header());
        Self(raw)
    }
}
impl Into<OpaqueTerm> for Header {
    fn into(self) -> OpaqueTerm {
        self.0
    }
}
