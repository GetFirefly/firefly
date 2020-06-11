use core::mem;
use core::ptr::NonNull;

#[cfg(target_pointer_width = "64")]
use heapless::consts::U152 as UHEAP_SIZES_LEN;
#[cfg(target_pointer_width = "32")]
use heapless::consts::U57 as UHEAP_SIZES_LEN;
use heapless::Vec;

use liblumen_alloc_macros::generate_heap_sizes;

use liblumen_core::alloc::mmap;
use liblumen_core::alloc::prelude::*;
use liblumen_core::alloc::size_classes::SizeClass;

use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::Term;
use crate::{SizeClassAlloc, SizeClassAllocRef};

/// This allocator is used to allocate process heaps globally.
///
/// It contains a reference to an instance of `StandardAlloc`
/// which is used to satisfy allocation requests.
pub struct ProcessHeapAlloc {
    alloc: SizeClassAllocRef,
    oversized_threshold: usize,
}
impl ProcessHeapAlloc {
    /// Size of word in bytes
    const WORD_SIZE: usize = mem::size_of::<usize>();

    // An array of heap sizes, using the same growth pattern as BEAM
    // Fibonnaci growth from 233 words, until 1M words, at which point
    // the growth increases 20% at a time
    generate_heap_sizes! {
        pub(super) const HEAP_SIZES: [usize; PLACEHOLDER] = [];
    }

    /// Corresponds to the first heap size of 233 words
    pub(super) const MIN_HEAP_SIZE_INDEX: usize = 0;

    /// Creates a new `ProcessAlloc` instance
    pub fn new() -> Self {
        let size_classes = &Self::HEAP_SIZES[..19]
            .iter()
            .map(|size| SizeClass::new(*size))
            .filter(|size_class| SizeClassAlloc::can_fit_multiple_blocks(size_class))
            .collect::<Vec<_, UHEAP_SIZES_LEN>>();
        let alloc = SizeClassAllocRef::new(&size_classes);
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
    pub fn alloc(&self, size: usize) -> AllocResult<*mut Term> {
        // Determine layout, require word alignment
        let layout = self.heap_layout(size);
        let total_size = layout.size();

        // Handle oversized heaps which need to be allocated using
        // the system allocator/mmap
        if total_size > self.oversized_threshold {
            return Self::alloc_oversized_heap(layout);
        }

        // Allocate region
        match unsafe {
            self.alloc
                .as_mut()
                .allocate(layout, AllocInit::Uninitialized)
                .map(|block| block.ptr)
        } {
            Ok(non_null) => {
                let ptr = non_null.as_ptr() as *mut Term;

                // Return pointer to the heap
                Ok(ptr)
            }
            Err(_) => Err(alloc!()),
        }
    }

    #[inline]
    fn alloc_oversized_heap(layout: Layout) -> AllocResult<*mut Term> {
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
    ) -> Result<*mut Term, AllocErr> {
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
            return Err(AllocErr);
        }

        let layout = self.heap_layout(size);
        let ptr = unsafe { NonNull::new_unchecked(heap as *mut u8) };

        if new_size < size {
            if let Ok(_) = unsafe {
                self.alloc
                    .as_mut()
                    .shrink(ptr, layout, new_size, ReallocPlacement::InPlace)
            } {
                return Ok(heap);
            }
        } else {
            if let Ok(_) = unsafe {
                self.alloc.as_mut().grow(
                    ptr,
                    layout,
                    new_size,
                    ReallocPlacement::InPlace,
                    AllocInit::Uninitialized,
                )
            } {
                return Ok(heap);
            }
        }

        Err(AllocErr)
    }

    /// Deallocate a process heap, releasing the memory back to the operating system
    pub unsafe fn dealloc(&self, heap: *mut Term, size: usize) {
        let layout = self.heap_layout(size);

        if layout.size() > self.oversized_threshold {
            // Deallocate oversized heap
            Self::dealloc_oversized_heap(heap, layout);
        } else {
            self.alloc
                .as_mut()
                .dealloc(NonNull::new_unchecked(heap as *mut u8), layout);
        }
    }

    pub(super) fn next_heap_size(size: usize) -> usize {
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
