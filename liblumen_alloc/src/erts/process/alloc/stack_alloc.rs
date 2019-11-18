use core::alloc::Layout;
use core::ops::DerefMut;
use core::ptr::NonNull;

use crate::erts;
use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::Term;

pub trait StackAlloc {
    /// Perform a stack allocation of `size` words to hold a single term.
    ///
    /// Returns `Err(Alloc)` if there is not enough space available
    ///
    /// NOTE: Do not use this to allocate space for multiple terms (lists
    /// and boxes count as a single term), as the size of the stack in terms
    /// is tied to allocations. Each time `stack_alloc` is called, the stack
    /// size is incremented by 1, and this enables efficient implementations
    /// of the other stack manipulation functions as the stack size in terms
    /// does not have to be recalculated constantly.
    unsafe fn alloca(&mut self, need: usize) -> AllocResult<NonNull<Term>>;

    /// Same as `alloca`, but does not validate that there is enough available space,
    /// as it is assumed that the caller has already validated those invariants
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.alloca(need).unwrap()
    }

    /// Perform a stack allocation, but with a `Layout`
    unsafe fn alloca_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        let need = erts::to_word_size(layout.size());
        self.alloca(need)
    }

    /// Same as `alloca_layout`, but does not validate that there is enough available space,
    /// as it is assumed that the caller has already validated those invariants
    unsafe fn alloca_layout_unchecked(&mut self, layout: Layout) -> NonNull<Term> {
        let need = erts::to_word_size(layout.size());
        self.alloca_unchecked(need)
    }
}
impl<A, S> StackAlloc for S
where
    A: StackAlloc,
    S: DerefMut<Target = A>,
{
    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloca(need)
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.deref_mut().alloca_unchecked(need)
    }

    #[inline]
    unsafe fn alloca_layout(&mut self, layout: Layout) -> AllocResult<NonNull<Term>> {
        self.deref_mut().alloca_layout(layout)
    }

    #[inline]
    unsafe fn alloca_layout_unchecked(&mut self, layout: Layout) -> NonNull<Term> {
        self.deref_mut().alloca_layout_unchecked(layout)
    }
}
