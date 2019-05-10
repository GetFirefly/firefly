use core::alloc::{AllocErr, Layout};
use core::mem::{self, ManuallyDrop};
use core::ptr::{self, NonNull};

use core::alloc::Alloc;

use crate::std_alloc::StandardAlloc;

/// This allocator is used to allocate process heaps globally.
///
/// It contains a reference to an instance of `StandardAlloc`
/// which is used to satisfy allocation requests.
#[repr(C)]
pub struct ProcessAlloc {
    alloc: ManuallyDrop<StandardAlloc>,
}
impl ProcessAlloc {
    // Size of word in bytes
    const WORD_SIZE: usize = mem::size_of::<usize>();

    // 233 words
    const DEFAULT_HEAP_SIZE: usize = 233 * Self::WORD_SIZE;

    /// Allocate a new default-sized process heap
    ///
    /// If this fails, it is because the system is out of memory,
    /// or due to a bug in the allocator framework
    #[inline(always)]
    pub fn alloc_default(&mut self) -> Result<*mut ProcessHeap, AllocErr> {
        self.alloc_heap(Self::DEFAULT_HEAP_SIZE)
    }

    /// Allocate a new heap of the given size
    ///
    /// If this fails, either there is an issue with the given size,
    /// the system is out of memory, or there is a bug in the allocator
    /// framework
    pub fn alloc_heap(&mut self, size: usize) -> Result<*mut ProcessHeap, AllocErr> {
        // Determine layout, require word alignment
        let header_layout = Layout::new::<ProcessHeap>();
        let layout = header_layout
            .extend_packed(Layout::from_size_align(size, Self::WORD_SIZE).unwrap())
            .unwrap();
        let total_size = layout.size();
        let header_size = header_layout.size();

        // Allocate region
        let ptr = unsafe { self.alloc.alloc(layout)?.as_ptr() };
        // Get heap pointer
        let heap_top = unsafe { ptr.offset(header_size as isize) };
        // Get stack pointer
        let stack_bottom = unsafe { ptr.offset((total_size as isize) - (header_size as isize)) };

        // Write header to beginning of region
        let header = ptr as *mut ProcessHeap;
        unsafe {
            ptr::write(
                header,
                ProcessHeap {
                    size: total_size,
                    heap_bottom: heap_top,
                    heap_top,
                    stack_bottom,
                    stack_top: stack_bottom,
                },
            );
        }

        // Return pointer to the heap
        Ok(header)
    }

    /// Reallocate a heap, attempting to keep it in place if possible,
    /// otherwise allocating a new heap and copying terms into the new heap
    ///
    /// TODO: This depends on having roots and GC, so that we can ensure that
    /// terms on the reallocated heap are updated so references point into the
    /// new heap, not the old
    pub unsafe fn realloc(
        &mut self,
        _heap: *mut ProcessHeap,
        _new_size: usize,
    ) -> Result<*mut ProcessHeap, AllocErr> {
        unimplemented!()
    }

    /// Deallocate a process heap, releasing the memory back to the operating system
    pub unsafe fn dealloc(&mut self, heap: *mut ProcessHeap) {
        let size = (*heap).size;
        let layout = Layout::from_size_align_unchecked(size, Self::WORD_SIZE);
        self.alloc
            .dealloc(NonNull::new_unchecked(heap as *mut u8), layout);
    }
}

#[repr(C)]
pub struct ProcessHeap {
    // The total size of this heap, including header
    size: usize,
    // Heap grows upwards, so as allocations are made
    // heap_top will grow towards stack_bottom.
    //
    // The heap_bottom field tells us where the heap begins.
    heap_top: *const u8,
    heap_bottom: *const u8,
    // Stack grows downwards, so as allocations are made
    // stack_bottom will grow towards heap_top
    //
    // The stack_top field tells us where the heap begins.
    stack_bottom: *const u8,
    stack_top: *const u8,
}

impl ProcessHeap {
    /// Perform a heap allocation
    pub unsafe fn malloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let size = layout.size();

        if self.available_heap() < size {
            // This will trigger a GC
            return Err(AllocErr);
        }

        let ptr = self.heap_top.offset(size as isize);
        Ok(NonNull::new_unchecked(ptr as *mut u8))
    }

    /// Perform a stack allocation
    pub unsafe fn alloca(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        let size = layout.size();

        if self.available_stack() < size {
            // This will trigger a GC
            return Err(AllocErr);
        }

        let ptr = self.stack_bottom.offset(-(size as isize));
        Ok(NonNull::new_unchecked(ptr as *mut u8))
    }

    /// Get the set of roots to follow for GC
    pub unsafe fn rootset(&mut self) -> *const RootSet {
        unimplemented!()
    }

    /// Returns the amount of usable space in this heap, regardless of allocations
    #[inline(always)]
    pub fn usable_size(&self) -> usize {
        (self.stack_top as usize) - (self.heap_bottom as usize)
    }

    /// Returns the amount of available stack space
    #[inline(always)]
    pub fn available_stack(&self) -> usize {
        (self.stack_bottom as usize) - (self.heap_top as usize)
    }

    /// Returns the amount of available stack space
    /// NOTE: This is the same calculation as `available_stack`, but describes intent
    #[inline(always)]
    pub fn available_heap(&self) -> usize {
        (self.stack_bottom as usize) - (self.heap_top as usize)
    }
}

/// A root set is a vector of pointers to live terms on the stack or heap
/// which need to be followed in order to perform garbage collection
pub type RootSet = *const u8;
