use core::alloc::{CannotReallocInPlace, Layout};
use core::mem;
use core::ptr::NonNull;

#[cfg(target_pointer_width = "64")]
use heapless::consts::U152 as UHEAP_SIZES_LEN;
#[cfg(target_pointer_width = "32")]
use heapless::consts::U57 as UHEAP_SIZES_LEN;
use heapless::Vec;

use lazy_static::lazy_static;

use liblumen_alloc_macros::generate_heap_sizes;

use liblumen_core::alloc::mmap;
use liblumen_core::alloc::size_classes::SizeClass;

use crate::erts::exception::system::Alloc;
use crate::erts::Term;
use crate::SizeClassAlloc;

// The global process heap allocator
lazy_static! {
    static ref PROC_ALLOC: ProcessHeapAlloc = ProcessHeapAlloc::new();
}

/// Allocate a new default sized process heap
#[inline]
pub fn default_heap() -> Result<(*mut Term, usize), Alloc> {
    let size = default_heap_size();
    PROC_ALLOC.alloc(size).map(|ptr| (ptr, size))
}

pub fn default_heap_size() -> usize {
    ProcessHeapAlloc::HEAP_SIZES[ProcessHeapAlloc::MIN_HEAP_SIZE_INDEX]
}

/// Allocate a new process heap of the given size
#[inline]
pub fn heap(size: usize) -> Result<*mut Term, Alloc> {
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
    let mut next_size = 0;
    for i in 0..ProcessHeapAlloc::HEAP_SIZES.len() {
        next_size = ProcessHeapAlloc::HEAP_SIZES[i];
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
pub struct ProcessHeapAlloc {
    alloc: SizeClassAlloc,
    oversized_threshold: usize,
}
impl ProcessHeapAlloc {
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
            .filter(|size_class| SizeClassAlloc::can_fit_multiple_blocks(size_class))
            .collect::<Vec<_, UHEAP_SIZES_LEN>>();
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
    pub fn alloc(&self, size: usize) -> Result<*mut Term, Alloc> {
        // Determine layout, require word alignment
        let layout = self.heap_layout(size);
        let total_size = layout.size();

        // Handle oversized heaps which need to be allocated using
        // the system allocator/mmap
        if total_size > self.oversized_threshold {
            return Self::alloc_oversized_heap(layout);
        }

        // Allocate region
        match unsafe { self.alloc.allocate(layout) } {
            Ok(non_null) => {
                let ptr = non_null.as_ptr() as *mut Term;

                // Return pointer to the heap
                Ok(ptr)
            }
            Err(_) => Err(alloc!()),
        }
    }

    #[inline]
    fn alloc_oversized_heap(layout: Layout) -> Result<*mut Term, Alloc> {
        match unsafe { mmap::map(layout) } {
            Ok(non_null) => {
                let ptr = non_null.as_ptr() as *mut Term;

                Ok(ptr)
            }
            Err(_) => Err(alloc!()),
        }
    }

    #[inline]
    pub fn realloc_in_place(
        &self,
        heap: *mut Term,
        size: usize,
        new_size: usize,
    ) -> Result<*mut Term, CannotReallocInPlace> {
        // Nothing to do if the size didn't change
        if size == new_size {
            return Ok(heap);
        }

        // For now we are not going to support shrinking via realloc_in_place of oversized heaps.
        // but we'll allow consumers of this API to believe that the realloc was successful,
        // this just means that there is now wastage of that unused space. Ideally we would
        // use mremap or its equivalent to handle this, but due to wide variance in support
        // and behaviour across platforms, it is easier now to just avoid shrinking. For growth,
        // consumers will need to do their own remapping by allocating a new heap, etc.
        if size > self.oversized_threshold {
            if new_size < size {
                return Ok(heap);
            }
            return Err(CannotReallocInPlace);
        }

        let layout = self.heap_layout(size);
        let ptr = unsafe { NonNull::new_unchecked(heap as *mut u8) };

        if let Ok(_) = unsafe { self.alloc.realloc_in_place(ptr, layout, new_size) } {
            return Ok(heap);
        }

        Err(CannotReallocInPlace)
    }

    /// Deallocate a process heap, releasing the memory back to the operating system
    pub unsafe fn dealloc(&self, heap: *mut Term, size: usize) {
        let layout = self.heap_layout(size);

        if layout.size() > self.oversized_threshold {
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
unsafe impl Send for ProcessHeapAlloc {}
unsafe impl Sync for ProcessHeapAlloc {}
