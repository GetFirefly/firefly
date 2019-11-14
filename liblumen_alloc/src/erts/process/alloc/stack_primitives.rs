use core::ops::DerefMut;

use crate::erts::term::prelude::Term;

pub trait StackPrimitives {
    /// Gets the number of terms currently allocated on the stack
    fn stack_size(&self) -> usize;

    /// Manually sets the stack size
    ///
    /// # Safety
    ///
    /// This is super unsafe, its only use is when constructing objects such as lists
    /// on the stack incrementally, thus representing a single logical term but composed
    /// of many small allocations. As `alloca_*` increments the `stack_size` value each
    /// time it is called, `set_stack_size` can be used to fix that value once construction
    /// is finished.
    unsafe fn set_stack_size(&mut self, size: usize);

    /// Returns the current stack pointer (pointer to the top of the stack)
    fn stack_pointer(&mut self) -> *mut Term;

    /// Manually sets the stack pointer to the given pointer
    ///
    /// NOTE: This will panic if the stack pointer is outside the process heap
    ///
    /// # Safety
    ///
    /// This is obviously super unsafe, but is useful as an optimization in some
    /// cases where a stack allocated object is being constructed but fails partway,
    /// and needs to be freed
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term);

    /// Gets the current amount of stack space used (in words)
    fn stack_used(&self) -> usize;

    /// Gets the current amount of space (in words) available for stack allocations
    fn stack_available(&self) -> usize;

    /// This function returns the term located in the given stack slot, if it exists.
    ///
    /// The stack slots are 1-indexed, where `1` is the top of the stack, or most recent
    /// allocation, and `S` is the bottom of the stack, or oldest allocation.
    ///
    /// If `S > stack_size`, then `None` is returned. Otherwise, `Some(Term)` is returned.
    fn stack_slot(&mut self, n: usize) -> Option<Term>;

    /// This function "pops" the last `n` terms from the stack, making that
    /// space available for new stack allocations.
    ///
    /// # Safety
    ///
    /// This function will panic if given a value `n` which exceeds the current
    /// number of terms allocated on the stack
    fn stack_popn(&mut self, n: usize);
}
impl<A, S> StackPrimitives for S
where
    A: StackPrimitives,
    S: DerefMut<Target = A>,
{
    #[inline]
    fn stack_size(&self) -> usize {
        self.deref().stack_size()
    }

    #[inline]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.deref_mut().set_stack_size(size)
    }

    #[inline]
    fn stack_pointer(&mut self) -> *mut Term {
        self.deref_mut().stack_pointer()
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        self.deref_mut().set_stack_pointer(sp);
    }

    #[inline]
    fn stack_used(&self) -> usize {
        self.deref().stack_used()
    }

    #[inline]
    fn stack_available(&self) -> usize {
        self.deref().stack_available()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        self.deref_mut().stack_slot(n)
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        self.deref_mut().stack_popn(n);
    }
}
