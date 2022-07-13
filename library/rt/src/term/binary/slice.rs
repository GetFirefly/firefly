use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ptr;
use core::str;

use liblumen_binary::{Bitstring, Selection};

use crate::term::OpaqueTerm;

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
    /// We give the selection static lifetime because we are managing the lifetime
    /// of the referenced data manually. The Rust borrow checker is of no help to
    /// us with most term data structures, due to their lifetimes being tied to a
    /// specific process heap, which can be swapped between at arbitrary points.
    /// However, our memory management strategy ensures that we never free memory
    /// that is referenced by live objects, so we are relying on that here.
    selection: Selection<'static>,
}
impl BitSlice {
    pub const TYPE_ID: TypeId = TypeId::of::<BitSlice>();

    #[inline]
    pub fn new(owner: OpaqueTerm, data: &[u8], bit_offset: u8, num_bits: usize) -> Self {
        let selection =
            Selection::new(data, 0, bit_offset, None, num_bits).expect("invalid selection");

        Self { owner, selection }
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
        let byte_size = ptr::metadata(self.data);
        f.debug_struct("BitSlice")
            .field("owner", &self.owner)
            .field("selection", &self.selection)
            .finish()
    }
}
impl Eq for BitSlice {}
impl<T: Bitstring> PartialEq<T> for BitSlice {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.selection.eq(other)
    }
}
impl Ord for BitSlice {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.selection.cmp(other)
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
