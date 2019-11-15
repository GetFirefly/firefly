mod heap;
mod iter;
mod process_heap_alloc;
mod semispace;
mod stack_alloc;
mod stack_primitives;
mod term_alloc;
mod virtual_alloc;
mod virtual_binary_heap;

pub use self::heap::{Heap, HeapAlloc};
pub use self::iter::HeapIter;
pub use self::process_heap_alloc::ProcessHeapAlloc;
pub use self::semispace::{GenerationalHeap, SemispaceHeap};
pub use self::stack_alloc::StackAlloc;
pub use self::stack_primitives::StackPrimitives;
pub use self::term_alloc::TermAlloc;
pub use self::virtual_alloc::{VirtualAlloc, VirtualAllocator, VirtualHeap};
pub use self::virtual_binary_heap::VirtualBinaryHeap;

use core::alloc::CannotReallocInPlace;

use lazy_static::lazy_static;

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::Term;

// The global process heap allocator
lazy_static! {
    static ref PROC_ALLOC: ProcessHeapAlloc = ProcessHeapAlloc::new();
}

/// Allocate a new default-sized process heap
#[inline]
pub fn default_heap() -> AllocResult<(*mut Term, usize)> {
    let size = default_heap_size();
    PROC_ALLOC.alloc(size).map(|ptr| (ptr, size))
}

/// Returns the default heap size for a process heap
pub fn default_heap_size() -> usize {
    ProcessHeapAlloc::HEAP_SIZES[ProcessHeapAlloc::MIN_HEAP_SIZE_INDEX]
}

/// Allocate a new process heap of the given size
#[inline]
pub fn heap(size: usize) -> AllocResult<*mut Term> {
    PROC_ALLOC.alloc(size)
}

/// Reallocate a process heap, in place
///
/// If reallocating and trying to grow the heap, if the allocation cannot be done
/// in place, then `Err(CannotReallocInPlace)` will be returned
#[inline]
pub unsafe fn realloc(
    heap: *mut Term,
    size: usize,
    new_size: usize,
) -> Result<*mut Term, CannotReallocInPlace> {
    PROC_ALLOC.realloc_in_place(heap, size, new_size)
}

/// Deallocate a heap previously allocated via `heap`
#[inline]
pub unsafe fn free(heap: *mut Term, size: usize) {
    PROC_ALLOC.dealloc(heap, size)
}

/// Calculates the next largest heap size equal to or greater than `size`
#[inline]
pub fn next_heap_size(size: usize) -> usize {
    ProcessHeapAlloc::next_heap_size(size)
}
