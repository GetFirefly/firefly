use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::NonNull;

use lazy_static::lazy_static;

use liblumen_alloc_macros::generate_heap_sizes;
use liblumen_core::alloc::mmap;
use liblumen_core::alloc::size_classes::SizeClass;

use crate::erts::Term;
use crate::SizeClassAlloc;

// The global process heap allocator
lazy_static! {
    static ref PROC_ALLOC: ProcessAlloc = ProcessAlloc::new();
}

/// Allocate a new process heap
///
/// If `size` is `None`, a default sized heap of 233 words will be allocated,
/// otherwise the provided size will be used.
#[inline]
pub fn heap(size: Option<usize>) -> Result<*mut Term, AllocErr> {
    match size {
        None => PROC_ALLOC.alloc(ProcessAlloc::HEAP_SIZES[ProcessAlloc::MIN_HEAP_SIZE_INDEX]),
        Some(size) => PROC_ALLOC.alloc(size),
    }
}

/// Reallocate a process heap, in place
///
/// If reallocating and trying to grow the heap, if the allocation cannot be done
/// in place, then `Err(AllocErr)` will be returned
#[inline]
pub unsafe fn realloc(
    heap: *mut Term,
    size: usize,
    new_size: usize,
) -> Result<*mut Term, AllocErr> {
    PROC_ALLOC.realloc(heap, size, new_size)
}

/// Deallocate a heap previously allocated via `heap`
#[inline]
pub unsafe fn free(heap: *mut Term, size: usize) {
    PROC_ALLOC.dealloc(heap, size)
}

/// Calculates the next largest heap size equal to or greater than `size`
#[inline]
pub fn next_heap_size(size: usize) -> usize {
    let mut next_size = 0;
    for i in 0..ProcessAlloc::HEAP_SIZES.len() {
        next_size = ProcessAlloc::HEAP_SIZES[i];
        if next_size > size {
            return next_size;
        }
    }
    // If we reach this point, we just grow by 20% until
    // we have a large enough region, these will be allocated
    // using `mmap` directly
    while size >= next_size {
        next_size = next_size + next_size / 5;
    }
    next_size
}

/// This allocator is used to allocate process heaps globally.
///
/// It contains a reference to an instance of `StandardAlloc`
/// which is used to satisfy allocation requests.
pub struct ProcessAlloc {
    alloc: SizeClassAlloc,
    oversized_threshold: usize,
}
impl ProcessAlloc {
    /// Size of word in bytes
    const WORD_SIZE: usize = mem::size_of::<usize>();

    /// An array of heap sizes, using the same growth pattern as BEAM
    /// Fibonnaci growth from 233 words, until 1M words, at which point
    /// the growth increases 20% at a time
    generate_heap_sizes! {
        const HEAP_SIZES: [usize; PLACEHOLDER] = [];
    }

    /// Corresponds to the first heap size of 233 words
    const MIN_HEAP_SIZE_INDEX: usize = 0;

    /// Creates a new `ProcessAlloc` instance
    pub fn new() -> Self {
        let size_classes = &Self::HEAP_SIZES[..19]
            .iter()
            .map(|size| SizeClass::new(*size))
            .collect::<Vec<_>>();
        let alloc = SizeClassAlloc::new(&size_classes);
        let oversized_threshold = alloc.max_size_class();
        Self {
            alloc,
            oversized_threshold,
        }
    }

    /// Allocate a new heap of the given size (in words)
    ///
    /// If this fails, either there is an issue with the given size,
    /// the system is out of memory, or there is a bug in the allocator
    /// framework
    pub fn alloc(&self, size: usize) -> Result<*mut Term, AllocErr> {
        // Determine layout, require word alignment
        let layout = self.heap_layout(size);
        let total_size = layout.size();
        // Handle oversized heaps which need to be allocated using
        // the system allocator/mmap
        if total_size > self.oversized_threshold {
            return Self::alloc_oversized_heap(layout);
        }
        // Allocate region
        let ptr = unsafe { self.alloc.allocate(layout)?.as_ptr() as *mut Term };
        // Return pointer to the heap
        Ok(ptr)
    }

    #[inline]
    fn alloc_oversized_heap(layout: Layout) -> Result<*mut Term, AllocErr> {
        let ptr = unsafe { mmap::map(layout)?.as_ptr() as *mut Term };
        Ok(ptr)
    }

    #[inline]
    pub fn realloc(
        &self,
        heap: *mut Term,
        size: usize,
        new_size: usize,
    ) -> Result<*mut Term, AllocErr> {
        // Nothing to do if the size didn't change
        if size == new_size {
            return Ok(heap);
        }
        // For now we are not going to support realloc_in_place of oversized heaps
        if size > self.oversized_threshold {
            return Err(AllocErr);
        }

        let layout = self.heap_layout(size);
        let ptr = unsafe { NonNull::new_unchecked(heap as *mut u8) };
        if let Ok(_) = unsafe { self.alloc.realloc_in_place(ptr, layout, new_size) } {
            return Ok(heap);
        }

        Err(AllocErr)
    }

    /// Deallocate a process heap, releasing the memory back to the operating system
    pub unsafe fn dealloc(&self, heap: *mut Term, size: usize) {
        let layout = self.heap_layout(size);
        if size > self.oversized_threshold {
            // Deallocate oversized heap
            Self::dealloc_oversized_heap(heap, layout);
        } else {
            self.alloc
                .deallocate(NonNull::new_unchecked(heap as *mut u8), layout);
        }
    }

    #[inline]
    unsafe fn dealloc_oversized_heap(heap: *mut Term, layout: Layout) {
        mmap::unmap(heap as *mut u8, layout);
    }

    #[inline]
    fn heap_layout(&self, size: usize) -> Layout {
        let size_class = Self::size_to_size_class(size);
        Layout::from_size_align(size_class * mem::size_of::<Term>(), Self::WORD_SIZE).unwrap()
    }

    #[inline]
    fn size_to_size_class(size: usize) -> usize {
        for size_class in Self::HEAP_SIZES.iter() {
            let size_class = *size_class;
            if size_class >= size {
                return size_class;
            }
        }
        size
    }
}
unsafe impl Send for ProcessAlloc {}
unsafe impl Sync for ProcessAlloc {}
