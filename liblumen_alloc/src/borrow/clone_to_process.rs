use core::alloc::Layout;
use core::mem;
use core::ptr::NonNull;

use crate::erts::exception::system::Alloc;
use crate::erts::{self, HeapAlloc, HeapFragment, ProcessControlBlock, Term};

/// This trait represents cloning, like `Clone`, but specifically
/// in the context of terms which need to be cloned into the heap
/// of a specific process, rather than using the global allocator.
///
/// In particular this is used for persistent data structures like
/// `HashMap` which use clone-on-write behavior internally for mutable
/// operations, e.g. `insert`. Rather than using `Clone` which would not
/// do the right thing, we instead implement this trait, and ensure that
/// those operations are provided a mutable reference to the current process
/// so that the clone is into the process heap, rather than the global heap
///
/// NOTE: You can implement both `CloneInProcess` and `Clone` for a type,
/// just be aware that any uses of `Clone` will allocate on the global heap
pub trait CloneToProcess {
    /// Returns boxed copy of this value, performing any heap allocations
    /// using the process heap of `process`, possibly using heap fragments if
    /// there is not enough space for the cloned value
    fn clone_to_process(&self, process: &ProcessControlBlock) -> Term {
        let mut heap = process.acquire_heap();
        match self.clone_to_heap(&mut heap) {
            Ok(term) => term,
            Err(_) => {
                drop(heap);
                let (term, mut frag) = self.clone_to_fragment().unwrap();
                process.attach_fragment(unsafe { frag.as_mut() });
                term
            }
        }
    }

    /// Returns boxed copy of this value, performing any heap allocations
    /// using the given heap. If cloning requires allocation that exceeds
    /// the amount of memory available, this returns `Err(Alloc)`, otherwise
    /// it returns `Ok(Term)`
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc>;

    /// Returns boxed copy of this value and the heap fragment it was allocated into
    ///
    /// If unable to allocate a heap fragment that fits this value, `Err(Alloc)` is returned
    fn clone_to_fragment(&self) -> Result<(Term, NonNull<HeapFragment>), Alloc> {
        let need = self.size_in_words();
        let layout = unsafe {
            let size = need * mem::size_of::<usize>();
            Layout::from_size_align_unchecked(size, mem::align_of::<Term>())
        };
        let mut frag = unsafe { HeapFragment::new(layout)? };
        let frag_ref = unsafe { frag.as_mut() };
        let term = self.clone_to_heap(frag_ref)?;
        Ok((term, frag))
    }

    /// Returns the size in words needed to allocate this value
    fn size_in_words(&self) -> usize {
        erts::to_word_size(mem::size_of_val(self))
    }
}
