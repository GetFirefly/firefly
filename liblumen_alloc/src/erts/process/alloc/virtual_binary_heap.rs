use core::mem;
use core::ptr;

use crate::erts::term::prelude::{Bitstring, Boxed, ProcBin};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink, UnsafeRef};

use super::{VirtualAllocator, VirtualHeap};

intrusive_adapter!(pub ProcBinAdapter = UnsafeRef<ProcBin>: ProcBin { link: LinkedListLink });

/// An implementation of `VirtualAlloc` and `VirtualHeap` for reference-counted
/// binaries, i.e. procbins.
pub struct VirtualBinaryHeap {
    bins: LinkedList<ProcBinAdapter>,
    size: usize,
    used: usize,
}
impl VirtualAllocator<ProcBin> for VirtualBinaryHeap {
    fn virtual_alloc(&mut self, ptr: Boxed<ProcBin>) {
        let full_byte_len = ptr.as_ref().full_byte_len();
        self.bins
            .push_front(unsafe { UnsafeRef::from_raw(ptr.as_ptr()) });
        self.used += full_byte_len;
    }

    fn virtual_free(&mut self, ptr: Boxed<ProcBin>) {
        let raw = ptr.as_ptr();
        debug_assert!(self.virtual_contains(raw));
        unsafe {
            self.unlink_raw(raw);
            ptr::drop_in_place(raw);
        }
    }

    fn virtual_pop(&mut self, ptr: Boxed<ProcBin>) -> ProcBin {
        // Obtain raw pointer
        let raw = ptr.as_ptr();
        debug_assert!(self.virtual_contains(raw));
        // Remove from list
        unsafe {
            self.unlink_raw(raw);
        }
        // Clone reference on to the stack
        // NOTE: this invalidates any pointers to the old location,
        // including the pointer we were given
        let bin_ref = ptr.as_ref();
        let bin = bin_ref.clone();
        unsafe {
            // Drop old value to ensure it is no longer used,
            // and that the reference count increment from the
            // clone operation is balanced out by dropping the
            // old reference
            ptr::drop_in_place(raw);
        }
        // Decrement the heap size
        let full_byte_len = bin.full_byte_len();
        self.used -= full_byte_len;
        // Return the raw ProcBin
        bin
    }

    fn virtual_unlink(&mut self, ptr: Boxed<ProcBin>) {
        let raw = ptr.as_ptr();
        debug_assert!(self.virtual_contains(raw));
        // Decrement heap usage
        let bin_size = ptr.as_ref().full_byte_len();
        // Perform unlink
        unsafe { self.unlink_raw(raw) };
        // Update usage
        self.used -= bin_size;
    }

    fn virtual_contains<T: ?Sized>(&self, ptr: *const T) -> bool {
        self.bins
            .iter()
            .any(|bin_ref| ptr as *const () == bin_ref as *const _ as *const ())
    }

    unsafe fn virtual_clear(&mut self) {
        let mut cursor = self.bins.front_mut();
        while let Some(_binary) = cursor.get() {
            let ptr = cursor.remove().unwrap();
            ptr::drop_in_place(UnsafeRef::into_raw(ptr));
        }
    }
}
impl VirtualHeap<ProcBin> for VirtualBinaryHeap {
    #[inline]
    fn virtual_size(&self) -> usize {
        self.size
    }

    #[inline]
    fn virtual_heap_used(&self) -> usize {
        self.used
    }

    #[inline]
    fn virtual_heap_unused(&self) -> usize {
        // We don't actually refuse allocations on the virtual heap, but we
        // wait until the next should_collect occurs before increasing the
        // size of the virtual heap, as it will ensure a GC is performed, and
        // we want to wait until we've collected any binaries on the virtual
        // heap before increasing the size
        if self.size >= self.used {
            self.size - self.used
        } else {
            0
        }
    }
}
impl VirtualBinaryHeap {
    /// Create a new virtual heap with the given virtual heap size (in words)
    pub fn new(size: usize) -> Self {
        Self {
            bins: LinkedList::new(ProcBinAdapter::new()),
            size: size * mem::size_of::<usize>(),
            used: 0,
        }
    }

    #[inline]
    unsafe fn unlink_raw(&mut self, raw: *mut ProcBin) {
        // Remove from the list
        let mut cursor = self.bins.cursor_mut_from_ptr(raw);
        cursor.remove().unwrap();
    }
}
