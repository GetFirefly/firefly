pub mod alloc;
pub mod code;
mod flags;
mod gc;
mod heap;
mod mailbox;
mod priority;

use core::alloc::Layout;
use core::any::Any;
use core::cell::RefCell;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem;
use core::ops::DerefMut;
use core::ptr::{self, NonNull};
use core::str::Chars;
use core::sync::atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering};

use ::alloc::sync::Arc;

use hashbrown::{HashMap, HashSet};
use intrusive_collections::{LinkedList, UnsafeRef};

use liblumen_core::locks::{Mutex, MutexGuard, RwLock, SpinLock};

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::process::alloc::layout_to_words;
use crate::erts::term::{atom_unchecked, pid, reference, Atom, Integer, Pid, ProcBin};

use super::*;

pub use self::alloc::heap_alloc::{self, HeapAlloc};
pub use self::alloc::{
    default_heap, heap, next_heap_size, StackAlloc, StackPrimitives, VirtualAlloc,
};
use self::code::stack;
use self::code::stack::frame::{Frame, Placement};
pub use self::flags::*;
pub use self::flags::*;
use self::gc::{GcError, RootSet};
use self::heap::ProcessHeap;
pub use self::mailbox::*;
pub use self::priority::Priority;
use crate::erts::process::alloc::heap_alloc::MakePidError;
use crate::erts::process::code::Code;
use crate::erts::term::BytesFromBinaryError;

// 4000 in [BEAM](https://github.com/erlang/otp/blob/61ebe71042fce734a06382054690d240ab027409/erts/emulator/beam/erl_vm.h#L39)
cfg_if::cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
     pub const MAX_REDUCTIONS_PER_RUN: Reductions = 4_000;
  } else {
     pub const MAX_REDUCTIONS_PER_RUN: Reductions = 1_000;
  }
}

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
    /// ID of the scheduler that is running the process
    scheduler_id: Mutex<Option<scheduler::ID>>,
    /// The priority of the process in `scheduler`.
    pub priority: Priority,
    /// Process flags, e.g. `Process.flag/1`
    flags: AtomicProcessFlags,
    /// Minimum size of the heap that this process will start with
    min_heap_size: usize,
    /// The maximum size of the heap allowed for this process
    max_heap_size: usize,
    /// Minimum virtual heap size for this process
    min_vheap_size: usize,
    /// The percentage of used to unused space at which a collection is triggered
    gc_threshold: f64,
    /// The maximum number of minor collections before a full sweep occurs
    max_gen_gcs: usize,
    /// off-heap allocations
    off_heap: SpinLock<LinkedList<HeapFragmentAdapter>>,
    off_heap_size: AtomicUsize,
    /// Process dictionary
    dictionary: Mutex<HashMap<Term, Term>>,
    /// The `pid` of the process that `spawn`ed this process.
    parent_pid: Option<Pid>,
    pid: Pid,
    #[allow(dead_code)]
    initial_module_function_arity: Arc<ModuleFunctionArity>,
    /// The number of reductions in the current `run`.  `code` MUST return when `run_reductions`
    /// exceeds `MAX_REDUCTIONS_PER_RUN`.
    run_reductions: AtomicU16,
    pub total_reductions: AtomicU64,
    code_stack: Mutex<code::stack::Stack>,
    pub status: RwLock<Status>,
    pub registered_name: RwLock<Option<Atom>>,
    /// Pids of processes that are linked to this process and need to be exited when this process
    /// exits
    pub linked_pid_set: Mutex<HashSet<Pid>>,
    pub mailbox: Mutex<RefCell<Mailbox>>,
    // process heap, cache line aligned to avoid false sharing with rest of struct
    heap: Mutex<ProcessHeap>,
}
impl ProcessControlBlock {
    /// Creates a new PCB with a heap defined by the given pointer, and
    /// `heap_size`, which is the size of the heap in words.
    pub fn new(
        priority: Priority,
        parent_pid: Option<Pid>,
        initial_module_function_arity: Arc<ModuleFunctionArity>,
        heap: *mut Term,
        heap_size: usize,
    ) -> Self {
        let heap = ProcessHeap::new(heap, heap_size);
        let off_heap = SpinLock::new(LinkedList::new(HeapFragmentAdapter::new()));
        let pid = pid::next();

        Self {
            flags: AtomicProcessFlags::new(ProcessFlags::Default),
            min_heap_size: heap_size,
            max_heap_size: 0,
            min_vheap_size: 0,
            gc_threshold: 0.75,
            max_gen_gcs: 65535,
            off_heap,
            off_heap_size: AtomicUsize::new(0),
            dictionary: Default::default(),
            pid,
            status: Default::default(),
            mailbox: Default::default(),
            heap: Mutex::new(heap),
            code_stack: Default::default(),
            scheduler_id: Mutex::new(None),
            priority,
            parent_pid,
            initial_module_function_arity,
            run_reductions: Default::default(),
            total_reductions: Default::default(),
            registered_name: Default::default(),
            linked_pid_set: Default::default(),
        }
    }

    // Scheduler

    pub fn scheduler_id(&self) -> Option<scheduler::ID> {
        *self.scheduler_id.lock()
    }

    pub fn schedule_with(&self, scheduler_id: scheduler::ID) {
        *self.scheduler_id.lock() = Some(scheduler_id);
    }

    // Flags

    /// Set the given process flag
    #[inline]
    pub fn set_flags(&self, flags: ProcessFlags) {
        self.flags.set(flags);
    }

    /// Unset the given process flag
    #[inline]
    pub fn clear_flags(&self, flags: ProcessFlags) {
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
    pub unsafe fn alloc_nofrag(&self, need: usize) -> Result<NonNull<Term>, Alloc> {
        let mut heap = self.heap.lock();
        heap.alloc(need)
    }

    /// Same as `alloc_nofrag`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_nofrag_layout(&self, layout: Layout) -> Result<NonNull<Term>, Alloc> {
        let words = layout_to_words(layout);
        self.alloc_nofrag(words)
    }

    /// Skip allocating on the process heap and directly allocate a heap fragment
    #[inline]
    pub unsafe fn alloc_fragment(&self, need: usize) -> Result<NonNull<Term>, Alloc> {
        let layout = Layout::from_size_align_unchecked(
            need * mem::size_of::<Term>(),
            mem::align_of::<Term>(),
        );
        self.alloc_fragment_layout(layout)
    }

    /// Same as `alloc_fragment`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_fragment_layout(&self, layout: Layout) -> Result<NonNull<Term>, Alloc> {
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

    // Stack

    /// Frees stack space occupied by the last term on the stack,
    /// adjusting the stack pointer accordingly.
    ///
    /// Use `stack_popn` to pop multiple terms from the stack at once
    #[inline]
    pub fn stack_pop(&self) -> Option<Term> {
        let mut heap = self.heap.lock();
        match heap.stack_slot(1) {
            None => None,
            ok @ Some(_) => {
                heap.stack_popn(1);
                ok
            }
        }
    }

    unsafe fn alloca(&self, need: usize) -> Result<NonNull<Term>, Alloc> {
        let mut heap = self.heap.lock();
        heap.alloca(need)
    }

    /// Pushes an immediate term or reference to term/list on top of the stack.
    ///
    /// For boxed terms, the unboxed term needs to be allocated on the process and for non-empty
    /// lists both the head and tail needs to be allocated on the process.
    ///
    /// Returns `Err(Alloc)` if the process is out of stack space
    #[inline]
    pub fn stack_push(&self, term: Term) -> Result<(), Alloc> {
        assert!(term.is_runtime());
        unsafe {
            let stack0 = self.alloca(1)?.as_ptr();
            ptr::write(stack0, term);
        }
        Ok(())
    }

    /// Returns the term at the top of the stack
    #[inline]
    pub fn stack_top(&self) -> Option<Term> {
        let mut heap = self.heap.lock();
        heap.stack_slot(1)
    }

    pub fn stack_used(&self) -> usize {
        self.heap.lock().stack_used()
    }

    // Links

    pub fn link(&self, other: &ProcessControlBlock) {
        // link in order so that locks are always taken in the same order to prevent deadlocks
        if self.pid < other.pid {
            let mut self_pid_set = self.linked_pid_set.lock();
            let mut other_pid_set = other.linked_pid_set.lock();

            self_pid_set.insert(other.pid);
            other_pid_set.insert(self.pid);
        } else {
            other.link(self)
        }
    }

    pub fn unlink(&self, other: &ProcessControlBlock) {
        // unlink in order so that locks are always taken in the same order to prevent deadlocks
        if self.pid < other.pid {
            let mut self_pid_set = self.linked_pid_set.lock();
            let mut other_pid_set = other.linked_pid_set.lock();

            self_pid_set.remove(&other.pid);
            other_pid_set.remove(&self.pid);
        } else {
            other.unlink(self)
        }
    }

    // Pid

    pub fn pid(&self) -> Pid {
        self.pid
    }

    pub fn pid_term(&self) -> Term {
        unsafe { self.pid().as_term() }
    }

    // Send

    pub fn send_heap_message(&self, heap_fragment: NonNull<HeapFragment>, data: Term) {
        let heap_fragment_ptr = heap_fragment.as_ptr();

        let off_heap_unsafe_ref_heap_fragment = unsafe { UnsafeRef::from_raw(heap_fragment_ptr) };
        self.off_heap
            .lock()
            .push_back(off_heap_unsafe_ref_heap_fragment);

        let message_unsafe_ref_heap_fragment = unsafe { UnsafeRef::from_raw(heap_fragment_ptr) };

        self.send_message(Message::HeapFragment(message::HeapFragment {
            unsafe_ref_heap_fragment: message_unsafe_ref_heap_fragment,
            data,
        }));
    }

    pub fn send_from_self(&self, data: Term) {
        self.send_message(Message::Process(message::Process { data }));
    }

    /// Returns `true` if the process should stop waiting and be rescheduled as runnable.
    pub fn send_from_other(&self, data: Term) -> Result<bool, Alloc> {
        match self.heap.try_lock() {
            Some(ref mut destination_heap) => match data.clone_to_heap(destination_heap) {
                Ok(destination_data) => {
                    self.send_message(Message::Process(message::Process {
                        data: destination_data,
                    }));
                }
                Err(_) => {
                    let (heap_fragment_data, heap_fragment) = data.clone_to_fragment()?;

                    self.send_heap_message(heap_fragment, heap_fragment_data);
                }
            },
            None => {
                let (heap_fragment_data, heap_fragment) = data.clone_to_fragment()?;

                self.send_heap_message(heap_fragment, heap_fragment_data);
            }
        }

        // status.write() scope
        {
            let mut writable_status = self.status.write();

            if *writable_status == Status::Waiting {
                *writable_status = Status::Runnable;

                Ok(true)
            } else {
                Ok(false)
            }
        }
    }

    fn send_message(&self, message: Message) {
        self.mailbox.lock().borrow_mut().push(message)
    }

    // Terms

    pub fn binary_from_bytes(&self, bytes: &[u8]) -> Result<Term, Alloc> {
        self.acquire_heap().binary_from_bytes(bytes)
    }

    pub fn binary_from_str(&self, s: &str) -> Result<Term, Alloc> {
        self.acquire_heap().binary_from_str(s)
    }

    pub fn bytes_from_binary<'process>(
        &'process self,
        binary: Term,
    ) -> Result<&'process [u8], BytesFromBinaryError> {
        let mut heap_guard = self.acquire_heap();
        let heap: &'process mut ProcessHeap = unsafe {
            mem::transmute::<&'_ mut ProcessHeap, &'process mut ProcessHeap>(heap_guard.deref_mut())
        };

        heap.bytes_from_binary(binary)
    }

    pub fn charlist_from_str(&self, s: &str) -> Result<Term, Alloc> {
        self.acquire_heap().charlist_from_str(s)
    }

    pub fn closure(
        &self,
        creator: Term,
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
        env: Vec<Term>,
    ) -> Result<Term, Alloc> {
        self.acquire_heap()
            .closure(creator, module_function_arity, code, env)
    }

    /// Constructs a list of only the head and tail, and associated with the given process.
    pub fn cons(&self, head: Term, tail: Term) -> Result<Term, Alloc> {
        self.acquire_heap().cons(head, tail)
    }

    pub fn external_pid_with_node_id(
        &self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Term, MakePidError> {
        self.acquire_heap()
            .external_pid_with_node_id(node_id, number, serial)
    }

    pub fn float(&self, f: f64) -> Result<Term, Alloc> {
        self.acquire_heap().float(f)
    }

    pub fn integer<I: Into<Integer>>(&self, i: I) -> Result<Term, Alloc> {
        self.acquire_heap().integer(i)
    }

    pub fn list_from_chars(&self, chars: Chars) -> Result<Term, Alloc> {
        self.acquire_heap().list_from_chars(chars)
    }

    pub fn list_from_iter<I>(&self, iter: I) -> Result<Term, Alloc>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.acquire_heap().list_from_iter(iter)
    }

    pub fn list_from_slice(&self, slice: &[Term]) -> Result<Term, Alloc> {
        self.acquire_heap().list_from_slice(slice)
    }

    pub fn improper_list_from_iter<I>(&self, iter: I, last: Term) -> Result<Term, Alloc>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.acquire_heap().improper_list_from_iter(iter, last)
    }

    pub fn improper_list_from_slice(&self, slice: &[Term], tail: Term) -> Result<Term, Alloc> {
        self.acquire_heap().improper_list_from_slice(slice, tail)
    }

    pub fn map_from_slice(&self, slice: &[(Term, Term)]) -> Result<Term, Alloc> {
        self.acquire_heap().map_from_slice(slice)
    }

    pub fn pid_with_node_id(
        &self,
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Term, MakePidError> {
        self.acquire_heap()
            .pid_with_node_id(node_id, number, serial)
    }

    pub fn reference(&self, number: reference::Number) -> Result<Term, Alloc> {
        self.reference_from_scheduler(self.scheduler_id.lock().unwrap(), number)
    }

    pub fn reference_from_scheduler(
        &self,
        scheduler_id: scheduler::ID,
        number: reference::Number,
    ) -> Result<Term, Alloc> {
        self.acquire_heap().reference(scheduler_id, number)
    }

    pub fn resource(&self, value: Box<dyn Any>) -> Result<Term, Alloc> {
        self.acquire_heap().resource(value)
    }

    pub fn subbinary_from_original(
        &self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> Result<Term, Alloc> {
        self.acquire_heap().subbinary_from_original(
            original,
            byte_offset,
            bit_offset,
            full_byte_len,
            partial_byte_bit_len,
        )
    }

    pub fn tuple_from_iter<I>(&self, iterator: I, len: usize) -> Result<Term, Alloc>
    where
        I: Iterator<Item = Term>,
    {
        self.acquire_heap().tuple_from_iter(iterator, len)
    }

    pub fn tuple_from_slice(&self, slice: &[Term]) -> Result<Term, Alloc> {
        self.acquire_heap().tuple_from_slice(slice)
    }

    pub fn tuple_from_slices(&self, slices: &[&[Term]]) -> Result<Term, Alloc> {
        self.acquire_heap().tuple_from_slices(slices)
    }

    // Process Dictionary

    /// Puts a new value under the given key in the process dictionary
    pub fn put(&self, key: Term, value: Term) -> Result<Term, Alloc> {
        assert!(key.is_runtime(), "invalid key term for process dictionary");
        assert!(
            value.is_runtime(),
            "invalid value term for process dictionary"
        );

        // hold heap lock before dictionary lock
        let mut heap = self.acquire_heap();

        let heap_key = if key.is_immediate() {
            key
        } else {
            key.clone_to_heap(&mut heap)?
        };
        let heap_value = if value.is_immediate() {
            value
        } else {
            value.clone_to_heap(&mut heap)?
        };

        match self.dictionary.lock().insert(heap_key, heap_value) {
            None => Ok(Term::NIL),
            Some(old_value) => Ok(old_value),
        }
    }

    /// Gets a value from the process dictionary using the given key
    pub fn get(&self, key: Term) -> Term {
        assert!(key.is_runtime(), "invalid key term for process dictionary");

        match self.dictionary.lock().get(&key) {
            None => Term::NIL,
            // We can simply copy the term value here, since we know it
            // is either an immediate, or already located on the process
            // heap or in a heap fragment.
            Some(value) => *value,
        }
    }

    /// Deletes a key/value pair from the process dictionary
    pub fn delete(&self, key: Term) -> Term {
        assert!(key.is_runtime(), "invalid key term for process dictionary");

        match self.dictionary.lock().remove(&key) {
            None => Term::NIL,
            Some(old_value) => old_value,
        }
    }

    // Garbage Collection

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
        self.flags.are_set(ProcessFlags::ForceGC)
    }

    #[inline(always)]
    fn is_gc_delayed(&self) -> bool {
        self.flags.are_set(ProcessFlags::DelayGC)
    }

    #[inline(always)]
    fn is_gc_disabled(&self) -> bool {
        self.flags.are_set(ProcessFlags::DisableGC)
    }

    #[inline(always)]
    fn needs_fullsweep(&self) -> bool {
        self.flags.are_set(ProcessFlags::NeedFullSweep)
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
    pub fn garbage_collect(&self, need: usize, roots: &[Term]) -> Result<usize, GcError> {
        let mut heap = self.heap.lock();
        // The roots passed in here are pointers to the native stack/registers, all other roots
        // we are able to pick up from the current process context
        let mut rootset = RootSet::new(roots);
        // The process dictionary is also used for roots
        for (k, v) in self.dictionary.lock().iter() {
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

    // Running

    pub fn reduce(&self) {
        self.run_reductions.fetch_add(1, Ordering::SeqCst);
    }

    pub fn is_reduced(&self) -> bool {
        MAX_REDUCTIONS_PER_RUN <= self.run_reductions.load(Ordering::SeqCst)
    }

    /// Run process until `reductions` exceeds `MAX_REDUCTIONS` or process exits
    pub fn run(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
        arc_process.start_running();

        // `code` is expected to set `code` before it returns to be the next spot to continue
        let option_code = arc_process
            .code_stack
            .lock()
            .get(0)
            .map(|frame| frame.code());

        let code_result = match option_code {
            Some(code) => code(arc_process),
            None => Ok(arc_process.exit()),
        };

        arc_process.stop_running();

        code_result
    }

    fn start_running(&self) {
        *self.status.write() = Status::Running;
    }

    fn stop_running(&self) {
        self.total_reductions.fetch_add(
            self.run_reductions.load(Ordering::SeqCst) as u64,
            Ordering::SeqCst,
        );
        self.run_reductions.store(0, Ordering::SeqCst);

        let mut writable_status = self.status.write();

        if *writable_status == Status::Running {
            *writable_status = Status::Runnable
        }
    }

    /// Puts the process in the waiting status
    pub fn wait(&self) {
        *self.status.write() = Status::Waiting;
        self.run_reductions.fetch_add(1, Ordering::AcqRel);
    }

    pub fn exit(&self) {
        self.reduce();
        self.exception(exit!(atom_unchecked("normal")));
    }

    pub fn is_exiting(&self) -> bool {
        if let Status::Exiting(_) = *self.status.read() {
            true
        } else {
            false
        }
    }

    pub fn exception(&self, exception: runtime::Exception) {
        *self.status.write() = Status::Exiting(exception);
    }

    // Code Stack

    pub fn code_stack_len(&self) -> usize {
        self.code_stack.lock().len()
    }

    pub fn pop_code_stack(&self) {
        let mut locked_stack = self.code_stack.lock();
        locked_stack.pop().unwrap();
    }

    /// Calls top `Frame`'s `Code` if it exists and the process is not reduced.
    pub fn call_code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
        if !arc_process.is_reduced() {
            let option_code = arc_process
                .code_stack
                .lock()
                .get(0)
                .map(|frame| frame.code());

            match option_code {
                Some(code) => code(arc_process),
                None => Ok(()),
            }
        } else {
            Ok(())
        }
    }

    pub fn current_module_function_arity(&self) -> Option<Arc<ModuleFunctionArity>> {
        self.code_stack
            .lock()
            .get(0)
            .map(|frame| frame.module_function_arity())
    }

    pub fn place_frame(&self, frame: Frame, placement: Placement) {
        match placement {
            Placement::Replace => self.replace_frame(frame),
            Placement::Push => self.push_frame(frame),
        }
    }

    pub fn push_frame(&self, frame: Frame) {
        self.code_stack.lock().push(frame)
    }

    pub fn replace_frame(&self, frame: Frame) {
        let mut locked_code_stack = self.code_stack.lock();

        // unwrap to ensure there is a frame to replace
        locked_code_stack.pop().unwrap();

        locked_code_stack.push(frame);
    }

    pub fn remove_last_frame(&self) {
        let mut locked_code_stack = self.code_stack.lock();

        assert_eq!(locked_code_stack.len(), 1);

        // unwrap to ensure there is a frame to replace
        locked_code_stack.pop().unwrap();
    }

    pub fn return_from_call(&self, term: Term) -> Result<(), Alloc> {
        let has_caller = {
            let mut locked_stack = self.code_stack.lock();

            // remove current frame.  The caller becomes the top frame, so it's
            // `module_function_arity` will be returned from
            // `current_module_function_arity`.
            locked_stack.pop();

            0 < locked_stack.len()
        };

        if has_caller {
            self.stack_push(term)
        } else {
            // no caller, return value is thrown away, process will exit when `Scheduler.run_once`
            // detects it has no frames.
            Ok(())
        }
    }

    pub fn stacktrace(&self) -> stack::Trace {
        self.code_stack.lock().trace()
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

impl fmt::Debug for ProcessControlBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.pid)?;

        match *self.registered_name.read() {
            Some(registered_name) => write!(f, " ({:?})", registered_name),
            None => Ok(()),
        }
    }
}

impl fmt::Display for ProcessControlBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let pid = self.pid;
        let (number, serial) = (pid.number(), pid.serial());

        write!(f, "#PID<0.{}.{}>", number, serial)?;

        match *self.registered_name.read() {
            Some(registered_name_atom) => write!(f, "({})", registered_name_atom.name()),
            None => Ok(()),
        }
    }
}

impl Eq for ProcessControlBlock {}

impl Hash for ProcessControlBlock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pid.hash(state);
    }
}

impl PartialEq for ProcessControlBlock {
    fn eq(&self, other: &Self) -> bool {
        self.pid == other.pid
    }
}

unsafe impl Send for ProcessControlBlock {}
unsafe impl Sync for ProcessControlBlock {}

type Reductions = u16;

// [BEAM statuses](https://github.com/erlang/otp/blob/551d03fe8232a66daf1c9a106194aa38ef660ef6/erts/emulator/beam/erl_process.c#L8944-L8972)
#[derive(Debug, PartialEq)]
pub enum Status {
    Runnable,
    Running,
    Waiting,
    Exiting(runtime::Exception),
}

impl Default for Status {
    fn default() -> Status {
        Status::Runnable
    }
}

#[cfg(test)]
mod test;
