#![allow(unused)]
use core::mem;
use core::ptr;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink, UnsafeRef};

use crate::erts::*;
use super::{YoungHeap, OldHeap};

intrusive_adapter!(pub ProcBinAdapter = UnsafeRef<ProcBin>: ProcBin { link: LinkedListLink });

pub struct VirtualBinaryHeap {
    bins: LinkedList<ProcBinAdapter>,
    size: usize,
    used: usize,
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

    /// Gets the current amount of virtual binary heap space used (in bytes)
    /// by binaries referenced from the current process
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Gets the current amount of virtual binary heap space used (in bytes)
    /// by binaries referenced from the current process
    #[inline]
    pub fn heap_used(&self) -> usize {
        self.used
    }

    /// Gets the current amount of "unused" virtual binary heap space (in bytes)
    ///
    /// This is a bit of a misnomer, since there isn't a real heap here, but we
    /// use this to drive decisions about when to perform a collection, like we
    /// do with the old/young heaps
    #[inline]
    pub fn unused(&self) -> usize {
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

    /// Returns true if the given pointer belongs to a binary on the virtual heap
    #[inline]
    pub fn contains<T>(&self, ptr: *const T) -> bool {
        self.bins.iter().any(|bin_ref| ptr == bin_ref as *const _ as *const T)
    }

    /// Adds the given `ProcBin` to the virtual binary heap
    ///
    /// Returns a box `Term` which wraps the pointer to the binary,
    /// and should be placed somewhere on the process heap to ensure
    /// the binary is not leaked
    #[inline]
    pub fn push(&mut self, bin: &ProcBin) -> Term {
        let term = unsafe { bin.as_term() };
        let size = bin.size();
        self.bins
            .push_front(unsafe { UnsafeRef::from_raw(bin as *const _ as *mut ProcBin) });
        self.used += size;
        term
    }

    /// Removes the pointed-to `ProcBin` from the virtual binary heap
    ///
    /// Returns the `ProcBin` indicated, which can either be dropped,
    /// or placed on a new virtual heap, whichever is desired. Note that
    ///
    /// NOTE: This operation is intended to mirror `push`, do not
    /// use it under any other circumstances
    #[inline]
    pub fn pop(&mut self, bin: *mut ProcBin) -> ProcBin {
        // Remove from the list
        let mut cursor = unsafe { self.bins.cursor_mut_from_ptr(bin) };
        let raw = cursor.remove().unwrap();
        let ptr = UnsafeRef::into_raw(raw);
        // Copy the header
        let bin = unsafe { ptr::read(ptr) };
        // Write the none value to the old location to ensure it is not used
        unsafe {
            ptr::write(ptr as *mut Term, Term::NONE);
        }
        // Decrement the heap size
        let bin_size = bin.size();
        self.used -= bin_size;
        // Return the raw ProcBin
        bin
    }

    /// Collect all binaries that are not located in `new_heap`
    #[inline]
    pub fn full_sweep(&mut self, new_heap: &YoungHeap) {
        let mut cursor = self.bins.front_mut();
        while let Some(binary) = cursor.get() {
            if !new_heap.contains(binary as *const _ as *const Term) {
                // This binary is no longer live, unlink it and drop it
                let ptr = cursor.remove().unwrap();
                unsafe { ptr::drop_in_place(UnsafeRef::into_raw(ptr)) };
            } else {
                cursor.move_next();
            }
        }
    }

    /// Collect all binaries that are not located in either `new_heap` or `old_heap`
    #[inline]
    pub fn sweep(&mut self, new_heap: &YoungHeap, old_heap: &OldHeap) {
        let mut cursor = self.bins.front_mut();
        while let Some(binary) = cursor.get() {
            let bin = binary as *const _ as *const Term;
            if !new_heap.contains(bin) && !old_heap.contains(bin) {
                // This binary is no longer live, unlink it and drop it
                let ptr = cursor.remove().unwrap();
                unsafe { ptr::drop_in_place(UnsafeRef::into_raw(ptr)) };
            } else {
                cursor.move_next();
            }
        }
    }
}
