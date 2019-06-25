use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::NonNull;

use lazy_static::lazy_static;
use heapless::Vec;
#[cfg(target_pointer_width = "64")]
use heapless::consts::U152 as UHEAP_SIZES_LEN;
#[cfg(target_pointer_width = "32")]
use heapless::consts::U57 as UHEAP_SIZES_LEN;

use liblumen_alloc_macros::generate_heap_sizes;
use liblumen_core::alloc::mmap;
use liblumen_core::alloc::size_classes::SizeClass;

use crate::erts::{self, Term};
use crate::SizeClassAlloc;

// The global process heap allocator
lazy_static! {
    static ref PROC_ALLOC: ProcessAlloc = ProcessAlloc::new();
}

/// Allocate a new default sized process heap
#[inline]
pub fn default_heap() -> Result<(*mut Term, usize), AllocErr> {
    let size = ProcessAlloc::HEAP_SIZES[ProcessAlloc::MIN_HEAP_SIZE_INDEX];
    PROC_ALLOC.alloc(size).map(|ptr| (ptr, size))
}

/// Allocate a new process heap of the given size
#[inline]
pub fn heap(size: usize) -> Result<*mut Term, AllocErr> {
    PROC_ALLOC.alloc(size)
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

/// A trait, like `Alloc`, specifically for allocation of terms on a process heap
pub trait AllocInProcess {
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

    /// Perform a stack allocation of `size` words to hold a single term.
    ///
    /// Returns `Err(AllocErr)` if there is not enough space available
    /// 
    /// NOTE: Do not use this to allocate space for multiple terms (lists
    /// and boxes count as a single term), as the size of the stack in terms
    /// is tied to allocations. Each time `stack_alloc` is called, the stack
    /// size is incremented by 1, and this enables efficient implementations
    /// of the other stack manipulation functions as the stack size in terms
    /// does not have to be recalculated constantly.
    unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr>;

    /// Same as `alloca`, but does not validate that there is enough available space,
    /// as it is assumed that the caller has already validated those invariants
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.alloca(need).unwrap()
    }

    /// Perform a stack allocation, but with a `Layout`
    unsafe fn alloca_layout(&mut self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        let need = erts::to_word_size(layout.size());
        self.alloca(need)
    }

    /// Same as `alloca_layout`, but does not validate that there is enough available space,
    /// as it is assumed that the caller has already validated those invariants
    unsafe fn alloca_layout_unchecked(&mut self, layout: Layout) -> NonNull<Term> {
        let need = erts::to_word_size(layout.size());
        self.alloca_unchecked(need)
    }

    /// Pushes a reference-counted binary on to this processes virtual heap
    ///
    /// NOTE: It is expected that the binary reference (the actual `ProcBin` struct)
    /// has already been allocated on the heap, and that this function is
    /// being called simply to add the reference to the virtual heap
    fn virtual_alloc(&mut self, bin: &erts::ProcBin) -> Term;

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
    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term>;

    /// This function "pops" the last `n` terms from the stack, making that
    /// space available for new stack allocations.
    ///
    /// # Safety
    ///
    /// This function will panic if given a value `n` which exceeds the current
    /// number of terms allocated on the stack
    #[inline]
    fn stack_popn(&mut self, n: usize);
}