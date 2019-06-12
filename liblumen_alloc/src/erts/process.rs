pub(crate) mod alloc;

use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::NonNull;

use intrusive_collections::{intrusive_adapter, UnsafeRef};
use intrusive_collections::{LinkedList, LinkedListLink};

use hashbrown::HashMap;

use super::*;

intrusive_adapter!(pub HeapFragmentAdapter = UnsafeRef<HeapFragment>: HeapFragment { link: LinkedListLink });

/// Represents the primary control structure for processes
#[repr(C)]
pub struct ProcessControlBlock {
    // Process flags, e.g. `Process.flag/1`
    pub(crate) flags: u32,
    // minimum size of the heap that this process will start with
    pub(crate) min_heap_size: usize,
    // the maximum size of the heap allowed for this process
    pub(crate) max_heap_size: usize,
    // minimum virtual heap size for this process
    pub(crate) min_vheap_size: usize,
    // the percentage of used to unused space at which a collection is triggered
    pub(crate) gc_threshold: f64,
    // the number of minor collections
    pub(crate) gen_gc_count: usize,
    // the maximum number of minor collections before a full sweep occurs
    pub(crate) max_gen_gcs: usize,
    // young generation heap
    pub(crate) young: YoungHeap,
    // old generation heap
    pub(crate) old: OldHeap,
    // virtual binary heap
    pub(crate) vheap: VirtualBinaryHeap,
    // off-heap allocations
    pub(crate) off_heap: LinkedList<HeapFragmentAdapter>,
    // process dictionary
    pub(crate) dictionary: HashMap<Term, Term>,
}
impl ProcessControlBlock {
    pub const FLAG_HEAP_GROW: u32 = 1 << 3;
    pub const FLAG_NEED_FULLSWEEP: u32 = 1 << 4;
    pub const FLAG_FORCE_GC: u32 = 1 << 10;
    pub const FLAG_DISABLE_GC: u32 = 1 << 11;
    pub const FLAG_DELAY_GC: u32 = 1 << 16;

    const DEFAULT_FLAGS: u32 = 0;

    /// Creates a new PCB with a heap defined by the given pointer, and
    /// `heap_size`, which is the size of the heap in words.
    #[inline]
    pub fn new(heap: *mut Term, heap_size: usize) -> Self {
        let young = YoungHeap::new(heap, heap_size);
        let old = OldHeap::default();
        let vheap = VirtualBinaryHeap::new();
        let off_heap = LinkedList::new(HeapFragmentAdapter::new());
        let dictionary = HashMap::new();
        Self {
            flags: Self::DEFAULT_FLAGS,
            min_heap_size: heap_size,
            max_heap_size: 0,
            min_vheap_size: 0,
            gc_threshold: 0.75,
            gen_gc_count: 0,
            max_gen_gcs: 0,
            young,
            old,
            vheap,
            off_heap,
            dictionary,
        }
    }

    /// Perform a heap allocation
    #[inline]
    pub unsafe fn malloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloc(need)
    }

    /// Perform an off-heap allocation using a `HeapFragment`
    #[inline]
    pub unsafe fn malloc_offheap(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let layout = Layout::from_size_align_unchecked(
            need * mem::size_of::<Term>(),
            mem::align_of::<Term>(),
        );
        let frag = HeapFragment::new(layout)?;
        let data = frag.as_ref().data().cast::<Term>();
        self.off_heap.push_front(UnsafeRef::from_raw(frag.as_ptr()));
        Ok(data)
    }

    /// Perform a stack allocation
    #[inline]
    pub unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloca(need)
    }

    /// Frees the last `words * mem::size_of::<Term>()` bytes on the stack
    #[inline]
    pub unsafe fn stack_pop(&mut self, words: usize) {
        self.young.stack_pop(words);
    }

    /// Determines if this heap should be collected
    #[inline]
    pub fn should_collect(&self) -> bool {
        let used = self.young.heap_used();
        let unused = self.young.unused();
        let threshold = ((used + unused) as f64 * self.gc_threshold).ceil() as usize;
        used >= threshold
    }

    /// Performs a garbage collection, using the provided root set
    ///
    /// The result is either `Ok(reductions)`, where `reductions` is the estimated cost
    /// of the collection in terms of reductions; or `Err(GcError)` containing the reason
    /// why garbage collection failed. Typically this will be due to attempting a minor
    /// collection and discovering that a full sweep is needed, rather than doing so automatically,
    /// the decision is left up to the caller to make. Other errors are described in the
    /// `GcError` documentation.
    #[inline]
    pub fn garbage_collect(&mut self, need: usize, roots: &[Term]) -> Result<usize, GcError> {
        // The roots passed in here are pointers to the native stack/registers, all other roots
        // we are able to pick up from the current process context
        let mut rootset = RootSet::new(roots);
        // The primary source of roots we add is the process stack
        rootset.push_range(self.young.stack_start, self.young.stack_used());
        // The process dictionary is also used for roots
        for val in self.dictionary.values() {
            rootset.push(val as *const _ as *mut _);
        }
        // Initialize the collector with the given root set
        let mut gc = GarbageCollector::new(self, rootset);
        // Run the collector
        gc.collect(need)
    }
}
