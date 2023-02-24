use alloc::alloc::Layout;
use core::assert_matches::debug_assert_matches;
use core::fmt;
use core::hash::{Hash, Hasher};

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;
use firefly_binary::{Bitstring, Selection};

use crate::gc::Gc;
use crate::term::{BinaryData, Boxable, Header, LayoutBuilder, OpaqueTerm, Tag, Term};

/// A slice of another binary or bitstring value
#[repr(C)]
pub struct BitSlice {
    header: Header,
    /// This a thin pointer to the original term we're borrowing from
    /// This is necessary to properly keep the owner live, either from the perspective
    /// of the garbage collector, or reference counting, until this slice is no
    /// longer needed.
    ///
    /// If the original data is not from a term, this will be None
    pub(crate) owner: OpaqueTerm,
    /// We give the selection static lifetime because we are managing the lifetime
    /// of the referenced data manually. The Rust borrow checker is of no help to
    /// us with most term data structures, due to their lifetimes being tied to a
    /// specific process heap, which can be swapped between at arbitrary points.
    /// However, our memory management strategy ensures that we never free memory
    /// that is referenced by live objects, so we are relying on that here.
    selection: Selection<'static>,
}
impl BitSlice {
    /// Create a new BitSlice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data` is a slice of bytes owned by `owner`. As such, we can
    /// guarantee that changing the lifetime to 'static is safe, since the data will not be dropped
    /// until the owning term has no more references. The garbage collector will take care of rewriting
    /// the pointer should the data be moved by a GC cycle.
    #[inline]
    pub unsafe fn new(owner: OpaqueTerm, data: &[u8], bit_offset: u8, num_bits: usize) -> Self {
        let data = core::mem::transmute::<_, &'static [u8]>(data);
        let selection =
            Selection::new(data, 0, bit_offset, None, num_bits).expect("invalid selection");

        Self {
            header: Header::new(Tag::Slice, 0),
            owner,
            selection,
        }
    }

    /// Returns the term from which this bit slice is derived
    #[inline]
    pub fn owner(&self) -> Term {
        self.owner.into()
    }

    /// Returns a raw pointer pointing to the owner term on a process heap
    ///
    /// This function may only be called when the owner of the underlying data
    /// is a heap binary, _not_ a ref-counted binary. No other term types are
    /// allowed for bit slices currently.
    ///
    /// This is only used internally for heap containment checks.
    pub(crate) unsafe fn owner_ptr(&self) -> *const () {
        let ptr = self.owner.as_ptr();
        debug_assert_matches!(self.owner(), Term::HeapBinary(_));
        ptr
    }

    /// Returns true if the term from which this bit slice is derived is a literal binary
    #[inline]
    pub fn is_owner_literal(&self) -> bool {
        self.owner.is_literal()
    }

    /// Returns true if the term from which this bit slice is derived is a ref-counted binary
    #[inline]
    pub fn is_owner_refcounted(&self) -> bool {
        self.owner.is_rc()
    }

    /// Create a BitSlice from an existing selection and its owning term
    #[inline]
    pub fn from_selection(owner: OpaqueTerm, selection: Selection<'static>) -> Self {
        Self {
            header: Header::new(Tag::Slice, 0),
            owner,
            selection,
        }
    }

    /// Returns the selection represented by this slice
    #[inline]
    pub fn as_selection(&self) -> Selection<'static> {
        self.selection
    }
}
impl Bitstring for BitSlice {
    #[inline]
    fn byte_size(&self) -> usize {
        self.selection.byte_size()
    }

    #[inline]
    fn bit_size(&self) -> usize {
        self.selection.bit_size()
    }

    #[inline]
    fn bit_offset(&self) -> u8 {
        self.selection.bit_offset()
    }

    #[inline]
    unsafe fn as_bytes_unchecked(&self) -> &[u8] {
        self.selection.as_bytes_unchecked()
    }
}
impl Clone for BitSlice {
    fn clone(&self) -> Self {
        let cloned = Self {
            header: Header::new(Tag::Slice, 0),
            owner: self.owner,
            selection: self.selection,
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
        f.debug_struct("BitSlice")
            .field("owner", &self.owner)
            .field("selection", &self.selection)
            .finish()
    }
}
impl Eq for BitSlice {}
impl crate::cmp::ExactEq for BitSlice {}
impl<T: Bitstring> PartialEq<T> for BitSlice {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.selection.eq(other)
    }
}
impl Ord for BitSlice {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.selection.cmp(&other.selection)
    }
}
impl<T: Bitstring> PartialOrd<T> for BitSlice {
    #[inline]
    fn partial_cmp(&self, other: &T) -> Option<core::cmp::Ordering> {
        self.selection.partial_cmp(other)
    }
}
impl Hash for BitSlice {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.selection.hash(state);
    }
}
impl fmt::Display for BitSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.selection)
    }
}
impl Boxable for BitSlice {
    type Metadata = ();

    const TAG: Tag = Tag::Slice;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains((self as *const Self).cast()) {
            return Layout::new::<()>();
        }
        let mut builder = LayoutBuilder::new();
        builder += Layout::new::<Self>();
        // If the referenced data is ref-counted, and is larger than a heap binary, its better to clone
        // the slice rather than the data. If the data is not ref-counted, or is a candidate for a heap
        // allocated binary, clone just the data referenced by the slice
        if !self.owner.is_literal() {
            let byte_size = self.byte_size();
            if byte_size <= BinaryData::MAX_HEAP_BYTES {
                builder.build_heap_binary(byte_size);
            }
        }
        builder.finish()
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            if self.owner.is_literal() {
                let mut cloned = Gc::new_uninit_in(heap).unwrap();
                unsafe {
                    self.write_clone_into_raw(cloned.as_mut_ptr());
                    return cloned.assume_init();
                }
            }

            if self.is_owner_refcounted() {
                self.owner.maybe_increment_refcount();
                let mut cloned = Gc::new_uninit_in(heap).unwrap();
                unsafe {
                    self.write_clone_into_raw(cloned.as_mut_ptr());
                    return cloned.assume_init();
                }
            }

            let byte_size = self.byte_size();
            assert!(byte_size <= BinaryData::MAX_HEAP_BYTES);
            let mut owner = Gc::<BinaryData>::with_capacity_in(byte_size, heap).unwrap();
            let selection = self.as_selection();
            let bit_size = selection.bit_size();
            owner.copy_from_selection(selection);
            let bytes = unsafe { owner.as_bytes_unchecked() };
            let clone = unsafe { Self::new(owner.into(), bytes, 0, bit_size) };
            Gc::new_in(clone, heap).unwrap()
        }
    }
}
