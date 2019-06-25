mod flags;
pub use self::flags::*;

mod alloc;
pub use self::alloc::{AllocInProcess, StackPrimitives};

mod gc;

use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use intrusive_collections::{LinkedList, UnsafeRef};
use liblumen_core::locks::SpinLock;

use self::gc::*;
use super::*;
use crate::borrow::CloneToProcess;

/// Represents the primary control structure for processes
#[repr(C)]
pub struct ProcessControlBlock {
    // Process flags, e.g. `Process.flag/1`
    flags: AtomicProcessFlag,
    // minimum size of the heap that this process will start with
    min_heap_size: usize,
    // the maximum size of the heap allowed for this process
    max_heap_size: usize,
    // minimum virtual heap size for this process
    min_vheap_size: usize,
    // the percentage of used to unused space at which a collection is triggered
    gc_threshold: f64,
    // the number of minor collections
    gen_gc_count: usize,
    // the maximum number of minor collections before a full sweep occurs
    max_gen_gcs: usize,
    // young generation heap
    young: YoungHeap,
    // old generation heap
    old: OldHeap,
    // off-heap allocations
    off_heap: SpinLock<LinkedList<HeapFragmentAdapter>>,
    off_heap_size: AtomicUsize,
    // process dictionary
    dictionary: HashMap<Term, Term>,
}
impl ProcessControlBlock {
    /// Creates a new PCB using default settings and heap size
    #[inline]
    pub fn default() -> Self {
        let (heap, heap_size) = alloc::default_heap().unwrap();
        Self::new(heap, heap_size)
    }

    /// Creates a new PCB with a heap defined by the given pointer, and
    /// `heap_size`, which is the size of the heap in words.
    #[inline]
    pub fn new(heap: *mut Term, heap_size: usize) -> Self {
        let young = YoungHeap::new(heap, heap_size);
        let old = OldHeap::default();
        let off_heap = SpinLock::new(LinkedList::new(HeapFragmentAdapter::new()));
        let dictionary = HashMap::new();
        Self {
            flags: AtomicProcessFlag::new(ProcessFlag::Default),
            min_heap_size: heap_size,
            max_heap_size: 0,
            min_vheap_size: 0,
            gc_threshold: 0.75,
            gen_gc_count: 0,
            max_gen_gcs: 65535,
            young,
            old,
            off_heap,
            off_heap_size: AtomicUsize::new(0),
            dictionary,
        }
    }

    /// Set the given process flag
    #[inline]
    pub fn set_flags(&self, flags: ProcessFlag) {
        self.flags.set(flags);
    }

    /// Unset the given process flag
    #[inline]
    pub fn clear_flags(&self, flags: ProcessFlag) {
        self.flags.clear(flags);
    }

    /// Perform a heap allocation, but do not fall back to allocating a heap fragment
    /// if the process heap is not able to fulfill the allocation request
    #[inline]
    pub unsafe fn alloc_nofrag(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloc(need)
    }

    /// Same as `alloc_nofrag`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_nofrag_layout(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<Term>, AllocErr> {
        let words = Self::layout_to_words(layout);
        self.alloc_nofrag(words)
    }

    /// Skip allocating on the process heap and directly allocate a heap fragment
    #[inline]
    pub unsafe fn alloc_fragment(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let layout = Layout::from_size_align_unchecked(
            need * mem::size_of::<Term>(),
            mem::align_of::<Term>(),
        );
        self.alloc_fragment_layout(layout)
    }

    /// Same as `alloc_fragment`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_fragment_layout(
        &mut self,
        layout: Layout,
    ) -> Result<NonNull<Term>, AllocErr> {
        let frag = HeapFragment::new(layout)?;
        let frag_ref = frag.as_ref();
        let size = frag_ref.size();
        let data = frag_ref.data().cast::<Term>();
        let mut off_heap = self.off_heap.lock();
        off_heap.push_front(UnsafeRef::from_raw(frag.as_ptr()));
        drop(off_heap);
        self.off_heap_size.fetch_add(size, Ordering::AcqRel);
        Ok(data)
    }

    /// Frees stack space occupied by the last term on the stack,
    /// adjusting the stack pointer accordingly.
    ///
    /// Use `stack_popn` to pop multiple terms from the stack at once
    #[inline]
    pub fn stack_pop(&mut self) -> Option<Term> {
        match self.stack_top() {
            None => None,
            ok @ Some(_) => {
                self.stack_popn(1);
                ok
            }
        }
    }

    /// Pushes an immediate term or reference to term/list on top of the stack
    ///
    /// Returns `Err(AllocErr)` if the process is out of stack space
    #[inline]
    pub fn stack_push(&mut self, term: Term) -> Result<(), AllocErr> {
        assert!(term.is_immediate() || term.is_boxed() || term.is_list());
        unsafe {
            let stack0 = self.alloca(1)?.as_ptr();
            ptr::write(stack0, term);
        }
        Ok(())
    }

    /// Returns the term at the top of the stack
    #[inline]
    pub fn stack_top(&mut self) -> Option<Term> {
        self.stack_slot(1)
    }

    /// Puts a new value under the given key in the process dictionary
    #[inline]
    pub fn put(&mut self, key: Term, value: Term) -> Term {
        assert!(
            key.is_immediate() || key.is_boxed() || key.is_list(),
            "invalid key term for process dictionary"
        );
        assert!(
            value.is_immediate() || value.is_boxed() || value.is_list(),
            "invalid value term for process dictionary"
        );

        let key = if key.is_immediate() {
            key
        } else {
            key.clone_to_process(self)
        };
        let value = if value.is_immediate() {
            value
        } else {
            value.clone_to_process(self)
        };

        match self.dictionary.insert(key, value) {
            None => Term::NIL,
            Some(old_value) => old_value,
        }
    }

    /// Gets a value from the process dictionary using the given key
    #[inline]
    pub fn get(&self, key: Term) -> Term {
        assert!(
            key.is_immediate() || key.is_boxed() || key.is_list(),
            "invalid key term for process dictionary"
        );

        match self.dictionary.get(&key) {
            None => Term::NIL,
            // We can simply copy the term value here, since we know it
            // is either an immediate, or already located on the process
            // heap or in a heap fragment.
            Some(value) => *value,
        }
    }

    /// Deletes a key/value pair from the process dictionary
    #[inline]
    pub fn delete(&mut self, key: Term) -> Term {
        assert!(
            key.is_immediate() || key.is_boxed() || key.is_list(),
            "invalid key term for process dictionary"
        );

        match self.dictionary.remove(&key) {
            None => Term::NIL,
            Some(old_value) => old_value,
        }
    }

    /// Determines if this heap should be collected
    ///
    /// NOTE: We require a mutable reference to self to call this,
    /// since only the owning scheduler should ever be initiating a collection
    #[inline]
    pub fn should_collect(&mut self) -> bool {
        // Check if a collection is being forced
        if self.is_gc_forced() {
            return true;
        }
        // Check if we definitely shouldn't collect
        if self.is_gc_delayed() || self.is_gc_disabled() {
            return false;
        }
        // Check if young generation requires collection
        let used = self.young.heap_used();
        let unused = self.young.unused();
        let threshold = ((used + unused) as f64 * self.gc_threshold).ceil() as usize;
        if used >= threshold {
            return true;
        }
        // Check if virtual heap size indicates we should do a collection
        let used = self.young.virtual_heap_used();
        let unused = self.young.virtual_heap_unused();
        if unused > 0 {
            let threshold = ((used + unused) as f64 * self.gc_threshold).ceil() as usize;
            used >= threshold
        } else {
            // We've exceeded the virtual heap size
            true
        }
    }

    #[inline(always)]
    fn off_heap_size(&self) -> usize {
        self.off_heap_size.load(Ordering::Acquire)
    }

    #[inline]
    fn is_gc_forced(&self) -> bool {
        self.flags.is_set(ProcessFlag::ForceGC)
    }

    #[inline(always)]
    fn is_gc_delayed(&self) -> bool {
        self.flags.is_set(ProcessFlag::DelayGC)
    }

    #[inline(always)]
    fn is_gc_disabled(&self) -> bool {
        self.flags.is_set(ProcessFlag::DisableGC)
    }

    #[inline(always)]
    fn needs_fullsweep(&self) -> bool {
        self.flags.is_set(ProcessFlag::NeedFullSweep)
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
        rootset.push_range(self.young.stack_pointer(), self.young.stack_size());
        // The process dictionary is also used for roots
        for (k, v) in &self.dictionary {
            rootset.push(k as *const _ as *mut _);
            rootset.push(v as *const _ as *mut _);
        }
        // Initialize the collector with the given root set
        let mut gc = GarbageCollector::new(self, rootset);
        // Run the collector
        gc.collect(need)
    }
}
impl Drop for ProcessControlBlock {
    fn drop(&mut self) {
        // The heap fragment list will be dropped after this, so we
        // do not need to worry about freeing those objects. Likewise,
        // the process dictionary should only carry immediates, or boxes
        // which point into the process heaps, so we do not need to free
        // any objects there ourselves, as they will be cleaned up when
        // the dictionary is dropped. This leaves only the young and old heap

        // Free young heap
        let young_heap_start = self.young.heap_start();
        let young_heap_size = self.young.size();
        unsafe { alloc::free(young_heap_start, young_heap_size) };
        // Free old heap, if active
        if self.old.active() {
            let old_heap_start = self.old.heap_start();
            let old_heap_size = self.old.size();
            unsafe { alloc::free(old_heap_start, old_heap_size) };
        }
    }
}
impl AllocInProcess for ProcessControlBlock {
    #[inline]
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        match self.young.alloc(need) {
            ok @ Ok(_) => ok,
            Err(_) => self.alloc_fragment(need),
        }
    }

    #[inline]
    unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        self.young.alloca(need)
    }

    #[inline]
    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        self.young.alloca_unchecked(need)
    }

    #[inline]
    fn virtual_alloc(&mut self, bin: &ProcBin) -> Term {
        self.young.virtual_alloc(bin)
    }

    #[inline]
    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        if self.young.contains(ptr) || self.old.contains(ptr) {
            return true;
        }
        if self.young.virtual_heap_contains(ptr) || self.old.virtual_heap_contains(ptr) {
            return true;
        }
        let off_heap = self.off_heap.lock();
        if off_heap.iter().any(|frag| frag.contains(ptr)) {
            return true;
        }
        false
    }
}
impl StackPrimitives for ProcessControlBlock {
    #[inline]
    fn stack_size(&self) -> usize {
        self.young.stack_size()
    }

    #[inline]
    unsafe fn set_stack_size(&mut self, size: usize) {
        self.young.set_stack_size(size);
    }

    #[inline]
    fn stack_pointer(&mut self) -> *mut Term {
        self.young.stack_pointer()
    }

    #[inline]
    unsafe fn set_stack_pointer(&mut self, sp: *mut Term) {
        self.young.set_stack_pointer(sp);
    }

    #[inline]
    fn stack_used(&self) -> usize {
        self.young.stack_used()
    }

    #[inline]
    fn stack_available(&self) -> usize {
        self.young.stack_available()
    }

    #[inline]
    fn stack_slot(&mut self, n: usize) -> Option<Term> {
        self.young.stack_slot(n)
    }

    #[inline]
    fn stack_popn(&mut self, n: usize) {
        self.young.stack_popn(n);
    }
}

#[cfg(test)]
mod test;
