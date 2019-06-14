#![allow(unused)]
use core::mem;
use core::ptr;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink, UnsafeRef};

use crate::erts::*;

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

    /// Like `size`, but in units of size `Term`
    ///
    /// NOTE: This is used when calculating whether to
    /// perform a garbage collection, as a large virtual binary heap
    /// indicates there is likely a considerable amount of memory that can
    /// be reclaimed by freeing references to binaries in the virtual
    /// heap
    #[inline]
    pub fn word_size(&self) -> usize {
        let bin_size = self.size();
        let bin_words = bin_size / mem::size_of::<Term>();
        let extra = bin_size % mem::size_of::<Term>();
        if extra > 0 {
            bin_words + 1
        } else {
            bin_words
        }
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
        unsafe { ptr::write(ptr as *mut Term, Term::NONE); }
        // Decrement the heap size
        let bin_size = bin.size();
        self.used -= bin_size;
        // Return the raw ProcBin
        bin
    }
}
