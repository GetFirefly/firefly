mod flags;
pub use self::flags::*;

mod alloc;
pub use self::alloc::{HeapAlloc, StackAlloc, StackPrimitives};

mod heap;
use self::heap::ProcessHeap;

mod gc;

use core::alloc::{AllocErr, Layout};
use core::mem;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicUsize, Ordering};

use hashbrown::HashMap;
use intrusive_collections::{LinkedList, UnsafeRef};
use liblumen_core::locks::{Mutex, MutexGuard, SpinLock};

use crate::borrow::CloneToProcess;

use self::gc::{GcError, RootSet};
use super::*;

/// Represents the primary control structure for processes
///
/// NOTE FOR LUKE: Like we discussed, when performing GC we will
/// note allow other threads to hold references to this struct, so
/// we need to wrap the acquisition of a process reference in an RwLock,
/// so that when the owning scheduler decides to perform GC, it can upgrade
/// to a writer lock and modify/access the heap, process dictionary, off-heap
/// fragments and message list without requiring multiple locks
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
    // the maximum number of minor collections before a full sweep occurs
    max_gen_gcs: usize,
    // off-heap allocations
    off_heap: SpinLock<LinkedList<HeapFragmentAdapter>>,
    off_heap_size: AtomicUsize,
    // process dictionary
    dictionary: HashMap<Term, Term>,
    // process heap, cache line aligned to avoid false sharing with rest of struct
    heap: Mutex<ProcessHeap>,
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
        let heap = ProcessHeap::new(heap, heap_size);
        let off_heap = SpinLock::new(LinkedList::new(HeapFragmentAdapter::new()));
        let dictionary = HashMap::new();
        Self {
            flags: AtomicProcessFlag::new(ProcessFlag::Default),
            min_heap_size: heap_size,
            max_heap_size: 0,
            min_vheap_size: 0,
            gc_threshold: 0.75,
            max_gen_gcs: 65535,
            off_heap,
            off_heap_size: AtomicUsize::new(0),
            dictionary,
            heap: Mutex::new(heap),
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

    /// Acquires exclusive access to the process heap, blocking the current thread until it is able
    /// to do so.
    ///
    /// The resulting lock guard can be used to perform multiple allocations without needing to
    /// acquire a lock multiple times. Once dropped, the lock is released.
    ///
    /// NOTE: This lock is re-entrant, so a single-thread may try to acquire a lock multiple times
    /// without deadlock, but in general you should acquire a lock with this function and then
    /// pass the guard into code which needs a lock, where possible.
    #[inline]
    pub fn acquire_heap<'a>(&'a self) -> MutexGuard<'a, ProcessHeap> {
        self.heap.lock()
    }

    /// Like `acquire_heap`, but instead of blocking the current thread when the lock is held by
    /// another thread, it returns `None`, allowing the caller to decide how to proceed. If lock
    /// acquisition is successful, it returns `Some(guard)`, which may be used in the same
    /// manner as `acquire_heap`.
    #[inline]
    pub fn try_acquire_heap<'a>(&'a self) -> Option<MutexGuard<'a, ProcessHeap>> {
        self.heap.try_lock()
    }

    /// Perform a heap allocation, but do not fall back to allocating a heap fragment
    /// if the process heap is not able to fulfill the allocation request
    #[inline]
    pub unsafe fn alloc_nofrag(&self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let mut heap = self.heap.lock();
        heap.alloc(need)
    }

    /// Same as `alloc_nofrag`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_nofrag_layout(&self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        let words = Self::layout_to_words(layout);
        self.alloc_nofrag(words)
    }

    /// Skip allocating on the process heap and directly allocate a heap fragment
    #[inline]
    pub unsafe fn alloc_fragment(&self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let layout = Layout::from_size_align_unchecked(
            need * mem::size_of::<Term>(),
            mem::align_of::<Term>(),
        );
        self.alloc_fragment_layout(layout)
    }

    /// Same as `alloc_fragment`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_fragment_layout(&self, layout: Layout) -> Result<NonNull<Term>, AllocErr> {
        let mut frag = HeapFragment::new(layout)?;
        let frag_ref = frag.as_mut();
        let data = frag_ref.data().cast::<Term>();
        self.attach_fragment(frag_ref);
        Ok(data)
    }

    /// Attaches a `HeapFragment` to this processes' off-heap fragment list
    #[inline]
    pub fn attach_fragment(&self, fragment: &mut HeapFragment) {
        let size = fragment.size();
        let mut off_heap = self.off_heap.lock();
        unsafe { off_heap.push_front(UnsafeRef::from_raw(fragment as *mut HeapFragment)) };
        drop(off_heap);
        self.off_heap_size.fetch_add(size, Ordering::AcqRel);
    }

    /// Attaches a ProcBin to this processes' virtual binary heap
    #[inline]
    pub fn virtual_alloc(&self, bin: &ProcBin) -> Term {
        let mut heap = self.heap.lock();
        heap.virtual_alloc(bin)
    }

    /// Frees stack space occupied by the last term on the stack,
    /// adjusting the stack pointer accordingly.
    ///
    /// Use `stack_popn` to pop multiple terms from the stack at once
    #[inline]
    pub fn stack_pop(&mut self) -> Option<Term> {
        let mut heap = self.heap.lock();
        match heap.stack_slot(1) {
            None => None,
            ok @ Some(_) => {
                heap.stack_popn(1);
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
        let mut heap = self.heap.lock();
        heap.stack_slot(1)
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
    pub fn should_collect(&self) -> bool {
        // Check if a collection is being forced
        if self.is_gc_forced() {
            return true;
        }
        // Check if we definitely shouldn't collect
        if self.is_gc_delayed() || self.is_gc_disabled() {
            return false;
        }
        // Check if young generation requires collection
        let heap = self.heap.lock();
        heap.should_collect(self.gc_threshold)
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
        let mut heap = self.heap.lock();
        // The roots passed in here are pointers to the native stack/registers, all other roots
        // we are able to pick up from the current process context
        let mut rootset = RootSet::new(roots);
        // The process dictionary is also used for roots
        for (k, v) in &self.dictionary {
            rootset.push(k as *const _ as *mut _);
            rootset.push(v as *const _ as *mut _);
        }
        // Initialize the collector with the given root set
        heap.garbage_collect(self, need, rootset)
    }

    /// Returns true if the given pointer belongs to memory owned by this process
    #[inline]
    pub fn is_owner<T>(&self, ptr: *const T) -> bool {
        let mut heap = self.heap.lock();
        if heap.is_owner(ptr) {
            return true;
        }
        drop(heap);
        let off_heap = self.off_heap.lock();
        if off_heap.iter().any(|frag| frag.contains(ptr)) {
            return true;
        }
        false
    }
}
#[cfg(test)]
impl ProcessControlBlock {
    #[inline]
    fn young_heap_used(&self) -> usize {
        let heap = self.heap.lock();
        heap.young.heap_used()
    }

    #[inline]
    fn old_heap_used(&self) -> usize {
        let heap = self.heap.lock();
        heap.old.heap_used()
    }

    #[inline]
    fn has_old_heap(&self) -> bool {
        let heap = self.heap.lock();
        heap.old.active()
    }
}
impl HeapAlloc for ProcessControlBlock {
    unsafe fn alloc(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let mut heap = self.heap.lock();
        heap.alloc(need)
    }

    fn is_owner<T>(&mut self, ptr: *const T) -> bool {
        let mut heap = self.heap.lock();
        heap.is_owner(ptr)
    }
}
impl StackAlloc for ProcessControlBlock {
    unsafe fn alloca(&mut self, need: usize) -> Result<NonNull<Term>, AllocErr> {
        let mut heap = self.heap.lock();
        heap.alloca(need)
    }

    unsafe fn alloca_unchecked(&mut self, need: usize) -> NonNull<Term> {
        let mut heap = self.heap.lock();
        heap.alloca_unchecked(need)
    }
}

#[cfg(test)]
mod test;
