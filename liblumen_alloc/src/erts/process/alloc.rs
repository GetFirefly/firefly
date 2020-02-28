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
use core::ptr;

use lazy_static::lazy_static;

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::Term;

pub const DEFAULT_STACK_SIZE: usize = 1; // 1 page
pub const STACK_ALIGNMENT: usize = 16;

// The global process heap allocator
lazy_static! {
    static ref PROC_ALLOC: ProcessHeapAlloc = ProcessHeapAlloc::new();
}

pub struct Stack {
    pub base: *mut u8,
    pub top: *mut u8,
    pub size: usize,
    pub end: *mut u8,
}
impl Stack {
    fn new(base: *mut u8, pages: usize) -> Self {
        use liblumen_core::alloc::utils::align_down_to;
        use liblumen_core::sys::sysconf;

        let page_size = sysconf::pagesize();
        let size = (pages + 1) * page_size;
        // Find the top of the stack
        let real_end = unsafe { base.offset(size as isize) };
        let end = unsafe { base.offset((pages * page_size) as isize) };
        let with_red_zone = unsafe { end.offset(-128) };
        let top = align_down_to(with_red_zone, STACK_ALIGNMENT);

        Self {
            base,
            top: top,
            size,
            end,
        }
    }

    #[inline]
    pub fn limit(&self) -> *mut u8 {
        self.end
    }
}
impl Default for Stack {
    fn default() -> Self {
        Self {
            base: ptr::null_mut(),
            top: ptr::null_mut(),
            size: 0,
            end: ptr::null_mut(),
        }
    }
}
/// This can be safely marked Sync when used in Process;
/// this is because the stack metadata is only ever accessed
/// by the executing process.
unsafe impl Sync for Stack {}

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

/// Allocate a new process stack of the given size
#[inline]
pub fn stack(size: usize) -> AllocResult<Stack> {
    use liblumen_core::alloc::mmap;

    debug_assert!(size > 0, "stack size in pages must be greater than 0");

    let ptr = unsafe { mmap::map_stack(size)? };
    Ok(Stack::new(ptr.as_ptr(), size))
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
