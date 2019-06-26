use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ops::DerefMut;
use core::ptr::NonNull;

use crate::erts::{self, Term};

/// A trait, like `Alloc`, specifically for allocation of terms on a process heap
pub trait HeapAlloc {
    /// Perform a heap allocation.
    ///
    /// If space on the process heap is not immediately available, then the allocation
    /// will be pushed into a heap fragment which will then be later moved on to the
    /// process heap during garbage collection
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr>;

    /// Same as `alloc`, but takes a `Layout` rather than the size in words
    unsafe fn alloc_layout(&mut self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        let need = erts::to_word_size(layout.size());
        self.alloc(need)
    }

    /*

        /// Pushes a reference-counted binary on to this processes virtual heap
        ///
        /// NOTE: It is expected that the binary reference (the actual `ProcBin` struct)
        /// has already been allocated on the heap, and that this function is
        /// being called simply to add the reference to the virtual heap
        fn virtual_alloc(&mut self, bin: &erts::ProcBin) -> Term;

    */
    /// Returns true if the given pointer is owned by this process/heap
    fn is_owner<T>(&mut self, ptr: *const T) -> bool;

    #[inline]
    fn layout_to_words(layout: Layout) -> usize {
        let size = layout.size();
        let mut words = size / mem::size_of::<Term>();
        if size % mem::size_of::<Term>() != 0 {
            words += 1;
        }
        words
    }
}
impl<A, H> HeapAlloc for H
where
    A: HeapAlloc,
    H: DerefMut<Target = A>,
{
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.deref_mut().alloc(need)
    }

    #[inline]
    unsafe fn alloc_layout(&mut self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        self.deref_mut().alloc_layout(layout)
    }

    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        self.deref_mut().is_owner(ptr)
    }
}
