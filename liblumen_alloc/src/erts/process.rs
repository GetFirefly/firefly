pub mod alloc;
pub mod code;
//pub mod ffi;
mod flags;
pub mod gc;
mod heap;
mod mailbox;
mod monitor;
pub mod priority;

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

use anyhow::*;
use hashbrown::{HashMap, HashSet};
use intrusive_collections::{LinkedList, UnsafeRef};

use liblumen_core::alloc::Layout;
use liblumen_core::locks::{Mutex, MutexGuard, RwLock, SpinLock};

use crate::borrow::CloneToProcess;
use crate::erts;
use crate::erts::exception::{AllocResult, ArcError, InternalResult, RuntimeException};
use crate::erts::module_function_arity::Arity;
use crate::erts::term::closure::{Creator, Definition, Index, OldUnique, Unique};
use crate::erts::term::prelude::*;

use super::*;

use self::alloc::VirtualAllocator;
use self::alloc::{Heap, HeapAlloc, TermAlloc};
use self::alloc::{StackAlloc, StackPrimitives};
use self::code::stack;
use self::code::stack::frame::{Frame, Placement};
use self::code::Code;
use self::gc::{GcError, RootSet};

pub use self::flags::*;
pub use self::heap::ProcessHeap;
pub use self::mailbox::*;
pub use self::monitor::Monitor;
pub use self::priority::Priority;

// 4000 in [BEAM](https://github.com/erlang/otp/blob/61ebe71042fce734a06382054690d240ab027409/erts/emulator/beam/erl_vm.h#L39)
cfg_if::cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
     pub const MAX_REDUCTIONS_PER_RUN: Reductions = 4_000;
  } else {
     pub const MAX_REDUCTIONS_PER_RUN: Reductions = 1_000;
  }
}

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct CalleeSavedRegisters {
    pub rsp: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbx: u64,
    pub rbp: u64,
}
/// NOTE: We can safely mark this Sync because
/// it is only ever used by the scheduler, and
/// is never accessed by other threads.
unsafe impl Sync for CalleeSavedRegisters {}

/// Represents the primary control structure for processes
///
/// NOTE FOR LUKE: Like we discussed, when performing GC we will
/// note allow other threads to hold references to this struct, so
/// we need to wrap the acquisition of a process reference in an RwLock,
/// so that when the owning scheduler decides to perform GC, it can upgrade
/// to a writer lock and modify/access the heap, process dictionary, off-heap
/// fragments and message list without requiring multiple locks
#[repr(C)]
pub struct Process {
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
    /// The `pid` of the process that does I/O on this process's behalf.
    group_leader_pid: Mutex<Pid>,
    pid: Pid,
    pub initial_module_function_arity: Arc<ModuleFunctionArity>,
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
    /// Maps monitor references to the PID of the process that is monitoring through that
    /// reference.
    pub monitor_by_reference: Mutex<HashMap<Reference, Monitor>>,
    /// Maps monitor references to the PID of the process being monitored by this process.
    pub monitored_pid_by_reference: Mutex<HashMap<Reference, Pid>>,
    pub mailbox: Mutex<RefCell<Mailbox>>,
    pub registers: CalleeSavedRegisters,
    pub stack: alloc::Stack,
    // process heap, cache line aligned to avoid false sharing with rest of struct
    heap: Mutex<ProcessHeap>,
}
impl Process {
    /// Creates a new PCB with a heap defined by the given pointer, and
    /// `heap_size`, which is the size of the heap in words.
    pub fn new(
        priority: Priority,
        parent: Option<&Self>,
        initial_module_function_arity: Arc<ModuleFunctionArity>,
        heap: *mut Term,
        heap_size: usize,
    ) -> Self {
        let heap = ProcessHeap::new(heap, heap_size);
        let off_heap = SpinLock::new(LinkedList::new(HeapFragmentAdapter::new()));
        let pid = Pid::next();

        // > When a new process is spawned, it gets the same group leader as the spawning process.
        // > Initially, at system startup, init is both its own group leader and the group leader
        // > of all processes.
        // > -- http://erlang.org/doc/man/erlang.html#group_leader-0
        let (parent_pid, group_leader_pid) = match parent {
            Some(parent) => (Some(parent.pid()), parent.get_group_leader_pid()),
            None => (None, pid),
        };

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
            stack: Default::default(),
            registers: Default::default(),
            code_stack: Default::default(),
            scheduler_id: Mutex::new(None),
            priority,
            parent_pid,
            group_leader_pid: Mutex::new(group_leader_pid),
            initial_module_function_arity,
            run_reductions: Default::default(),
            total_reductions: Default::default(),
            registered_name: Default::default(),
            linked_pid_set: Default::default(),
            monitor_by_reference: Default::default(),
            monitored_pid_by_reference: Default::default(),
        }
    }

    pub fn new_with_stack(
        priority: Priority,
        parent: Option<&Self>,
        initial_module_function_arity: Arc<ModuleFunctionArity>,
        heap: *mut Term,
        heap_size: usize,
    ) -> AllocResult<Self> {
        let mut p = Self::new(
            priority,
            parent,
            initial_module_function_arity,
            heap,
            heap_size,
        );
        p.stack = self::alloc::stack(4)?;
        Ok(p)
    }

    // Scheduler

    pub fn scheduler_id(&self) -> Option<scheduler::ID> {
        *self.scheduler_id.lock()
    }

    pub fn schedule_with(&self, scheduler_id: scheduler::ID) {
        *self.scheduler_id.lock() = Some(scheduler_id);
    }

    // Flags

    pub fn are_flags_set(&self, flags: ProcessFlags) -> bool {
        self.flags.are_set(flags)
    }

    /// Set the given process flags
    #[inline]
    pub fn set_flags(&self, flags: ProcessFlags) -> ProcessFlags {
        self.flags.set(flags)
    }

    /// Unset the given process flags
    #[inline]
    pub fn clear_flags(&self, flags: ProcessFlags) -> ProcessFlags {
        self.flags.clear(flags)
    }

    pub fn trap_exit(&self, value: bool) -> bool {
        let flag = ProcessFlags::TrapExit;

        let old_flags = if value {
            self.set_flags(flag)
        } else {
            self.clear_flags(flag)
        };

        old_flags.are_set(flag)
    }

    pub fn traps_exit(&self) -> bool {
        self.are_flags_set(ProcessFlags::TrapExit)
    }

    // Alloc

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
    pub unsafe fn alloc_nofrag(&self, need: usize) -> AllocResult<NonNull<Term>> {
        let mut heap = self.heap.lock();
        heap.deref_mut().alloc(need)
    }

    /// Same as `alloc_nofrag`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_nofrag_layout(&self, layout: Layout) -> AllocResult<NonNull<Term>> {
        let words = erts::to_word_size(layout.size());
        self.alloc_nofrag(words)
    }

    /// Skip allocating on the process heap and directly allocate a heap fragment
    #[inline]
    pub unsafe fn alloc_fragment(&self, need: usize) -> AllocResult<NonNull<Term>> {
        let layout = Layout::from_size_align_unchecked(
            need * mem::size_of::<Term>(),
            mem::align_of::<Term>(),
        );
        self.alloc_fragment_layout(layout)
    }

    /// Same as `alloc_fragment`, but takes a `Layout` rather than the size in words
    #[inline]
    pub unsafe fn alloc_fragment_layout(&self, layout: Layout) -> AllocResult<NonNull<Term>> {
        let mut frag = HeapFragment::new(layout)?;
        let frag_ref = frag.as_mut();
        let data = frag_ref.data().cast::<Term>();
        self.attach_fragment(frag_ref);
        Ok(data)
    }

    /// Attaches a `HeapFragment` to this processes' off-heap fragment list
    #[inline]
    pub fn attach_fragment(&self, fragment: &mut HeapFragment) {
        let size = fragment.heap_size();
        let mut off_heap = self.off_heap.lock();
        unsafe { off_heap.push_front(UnsafeRef::from_raw(fragment as *mut HeapFragment)) };
        drop(off_heap);
        self.off_heap_size.fetch_add(size, Ordering::AcqRel);
    }

    /// Attaches a ProcBin to this processes' virtual binary heap
    #[inline]
    pub fn virtual_alloc(&self, bin: &ProcBin) -> Term {
        let mut heap = self.heap.lock();
        let boxed: Boxed<ProcBin> = bin.into();
        heap.virtual_alloc(boxed);
        boxed.into()
    }

    // Stack

    /// Returns the nth (1-based) element from the top of the stack without removing it from the
    /// stack.
    pub fn stack_peek(&self, one_based_index: usize) -> Option<Term> {
        self.heap.lock().stack_slot(one_based_index)
    }

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

    /// Frees stack space occupied by the last `n` terms on the stack.
    ///
    /// Panics if the stack does not have that many items.
    pub fn stack_popn(&self, n: usize) {
        if n > 0 {
            self.heap.lock().stack_popn(n)
        }
    }

    unsafe fn alloca(&self, need: usize) -> AllocResult<NonNull<Term>> {
        let mut heap = self.heap.lock();
        heap.alloca(need)
    }

    pub unsafe fn alloca_layout(&self, layout: Layout) -> AllocResult<NonNull<Term>> {
        let mut heap = self.heap.lock();
        heap.alloca_layout(layout)
    }

    #[inline(always)]
    pub fn stack(&self) -> &alloc::Stack {
        &self.stack
    }

    /// Pushes an immediate term or reference to term/list on top of the stack.
    ///
    /// For boxed terms, the unboxed term needs to be allocated on the process and for non-empty
    /// lists both the head and tail needs to be allocated on the process.
    ///
    /// Returns `Err(Alloc)` if the process is out of stack space
    #[inline]
    pub fn stack_push(&self, term: Term) -> AllocResult<()> {
        assert!(term.is_valid());
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

    pub fn link(&self, other: &Process) {
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

    pub fn unlink(&self, other: &Process) {
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

    // Monitors

    pub fn monitor(&self, reference: Reference, monitored_pid: Pid) {
        self.monitored_pid_by_reference
            .lock()
            .insert(reference, monitored_pid);
    }

    pub fn demonitor(&self, reference: &Reference) -> Option<Pid> {
        self.monitored_pid_by_reference.lock().remove(reference)
    }

    pub fn monitored(&self, reference: Reference, monitor: Monitor) {
        self.monitor_by_reference.lock().insert(reference, monitor);
    }

    pub fn demonitored(&self, reference: &Reference) -> Option<Pid> {
        self.monitor_by_reference
            .lock()
            .remove(reference)
            .map(|monitor| *monitor.monitoring_pid())
    }

    // Group Leader Pid

    pub fn get_group_leader_pid(&self) -> Pid {
        *self.group_leader_pid.lock()
    }

    pub fn set_group_leader_pid(&self, group_leader_pid: Pid) {
        *self.group_leader_pid.lock() = group_leader_pid
    }

    pub fn get_group_leader_pid_term(&self) -> Term {
        self.get_group_leader_pid().encode().unwrap()
    }

    // Pid

    pub fn pid(&self) -> Pid {
        self.pid
    }

    pub fn pid_term(&self) -> Term {
        self.pid().encode().unwrap()
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
    pub fn send_from_other(&self, data: Term) -> AllocResult<bool> {
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

    pub fn binary_from_bytes(&self, bytes: &[u8]) -> AllocResult<Term> {
        self.acquire_heap().binary_from_bytes(bytes)
    }

    pub fn binary_from_str(&self, s: &str) -> AllocResult<Term> {
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

    pub fn charlist_from_str(&self, s: &str) -> AllocResult<Term> {
        self.acquire_heap()
            .charlist_from_str(s)
            .map(|list| list.into())
    }

    pub fn anonymous_closure_with_env_from_slice(
        &self,
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        code: Option<Code>,
        creator: Creator,
        slice: &[Term],
    ) -> AllocResult<Term> {
        self.acquire_heap()
            .anonymous_closure_with_env_from_slice(
                module, index, old_unique, unique, arity, code, creator, slice,
            )
            .map(|term_ptr| term_ptr.into())
    }

    pub fn export_closure(
        &self,
        module: Atom,
        function: Atom,
        arity: u8,
        code: Option<Code>,
    ) -> AllocResult<Term> {
        self.acquire_heap()
            .export_closure(module, function, arity, code)
            .map(|term_ptr| term_ptr.into())
    }

    /// Constructs a list of only the head and tail, and associated with the given process.
    pub fn cons(&self, head: Term, tail: Term) -> AllocResult<Term> {
        self.acquire_heap()
            .cons(head, tail)
            .map(|boxed| boxed.into())
    }

    pub fn external_pid(
        &self,
        node: Arc<Node>,
        number: usize,
        serial: usize,
    ) -> InternalResult<Term> {
        self.acquire_heap()
            .external_pid(node, number, serial)
            .map(|pid| pid.into())
    }

    pub fn float(&self, f: f64) -> AllocResult<Term> {
        self.acquire_heap().float(f).map(|f| f.into())
    }

    pub fn integer<I: Into<Integer>>(&self, i: I) -> AllocResult<Term> {
        self.acquire_heap().integer(i)
    }

    pub fn list_from_chars(&self, chars: Chars) -> AllocResult<Term> {
        self.acquire_heap()
            .list_from_chars(chars)
            .map(Self::optional_cons_to_term)
    }

    pub fn list_from_iter<I>(&self, iter: I) -> AllocResult<Term>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.acquire_heap()
            .list_from_iter(iter)
            .map(Self::optional_cons_to_term)
    }

    pub fn list_from_slice(&self, slice: &[Term]) -> AllocResult<Term> {
        self.acquire_heap()
            .list_from_slice(slice)
            .map(Self::optional_cons_to_term)
    }

    pub fn improper_list_from_iter<I>(&self, iter: I, last: Term) -> AllocResult<Term>
    where
        I: DoubleEndedIterator + Iterator<Item = Term>,
    {
        self.acquire_heap()
            .improper_list_from_iter(iter, last)
            .map(Self::optional_cons_to_term)
    }

    pub fn improper_list_from_slice(&self, slice: &[Term], tail: Term) -> AllocResult<Term> {
        self.acquire_heap()
            .improper_list_from_slice(slice, tail)
            .map(Self::optional_cons_to_term)
    }

    #[inline]
    fn optional_cons_to_term(cons: Option<Boxed<Cons>>) -> Term {
        match cons {
            None => Term::NIL,
            Some(boxed) => boxed.into(),
        }
    }

    pub fn map_from_hash_map(&self, hash_map: HashMap<Term, Term>) -> AllocResult<Term> {
        self.acquire_heap()
            .map_from_hash_map(hash_map)
            .map(|map| map.into())
    }

    pub fn map_from_slice(&self, slice: &[(Term, Term)]) -> AllocResult<Term> {
        self.acquire_heap()
            .map_from_slice(slice)
            .map(|map| map.into())
    }

    pub fn reference(&self, number: ReferenceNumber) -> AllocResult<Term> {
        self.reference_from_scheduler(self.scheduler_id.lock().unwrap(), number)
    }

    pub fn reference_from_scheduler(
        &self,
        scheduler_id: scheduler::ID,
        number: ReferenceNumber,
    ) -> AllocResult<Term> {
        self.acquire_heap()
            .reference(scheduler_id, number)
            .map(|ref_ptr| ref_ptr.into())
    }

    pub fn resource(&self, value: Box<dyn Any>) -> AllocResult<Term> {
        self.acquire_heap().resource(value).map(|r| r.into())
    }

    pub fn subbinary_from_original(
        &self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        full_byte_len: usize,
        partial_byte_bit_len: u8,
    ) -> AllocResult<Term> {
        self.acquire_heap()
            .subbinary_from_original(
                original,
                byte_offset,
                bit_offset,
                full_byte_len,
                partial_byte_bit_len,
            )
            .map(|sub_ptr| sub_ptr.into())
    }

    pub fn tuple_from_iter<I>(&self, iterator: I, len: usize) -> AllocResult<Term>
    where
        I: Iterator<Item = Term>,
    {
        self.acquire_heap()
            .tuple_from_iter(iterator, len)
            .map(|tup_ptr| tup_ptr.into())
    }

    pub fn tuple_from_slice(&self, slice: &[Term]) -> AllocResult<Term> {
        self.acquire_heap()
            .tuple_from_slice(slice)
            .map(|tup_ptr| tup_ptr.into())
    }

    pub fn tuple_from_slices(&self, slices: &[&[Term]]) -> AllocResult<Term> {
        self.acquire_heap()
            .tuple_from_slices(slices)
            .map(|tup_ptr| tup_ptr.into())
    }

    // Process Dictionary

    /// Puts a new value under the given key in the process dictionary
    pub fn put(&self, key: Term, value: Term) -> exception::Result<Term> {
        assert!(key.is_valid(), "invalid key term for process dictionary");
        assert!(
            value.is_valid(),
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
            None => Ok(atom!("undefined")),
            Some(old_value) => Ok(old_value),
        }
    }

    /// Gets a value from the process dictionary using the given key
    pub fn get_value_from_key(&self, key: Term) -> Term {
        assert!(key.is_valid(), "invalid key term for process dictionary");

        match self.dictionary.lock().get(&key) {
            None => atom!("undefined"),
            // We can simply copy the term value here, since we know it
            // is either an immediate, or already located on the process
            // heap or in a heap fragment.
            Some(value) => *value,
        }
    }

    /// Returns all key/value pairs from process dictionary
    pub fn get_entries(&self) -> AllocResult<Term> {
        let mut heap = self.heap.lock();
        let dictionary = self.dictionary.lock();

        let len = dictionary.len();
        let entry_need = Tuple::layout_for_len(2);
        let entry_need_in_words = erts::to_word_size(entry_need.size());
        let need_in_words = Cons::need_in_words_from_len(len) + len * entry_need_in_words;

        if need_in_words <= heap.heap_available() {
            let entry_vec: Vec<Term> = dictionary
                .iter()
                .map(|(key, value)| heap.tuple_from_slice(&[*key, *value]).unwrap().into())
                .collect();

            Ok(heap
                .list_from_slice(&entry_vec)
                .map(|list| list.into())
                .unwrap())
        } else {
            Err(alloc!())
        }
    }

    /// Returns list of all keys from the process dictionary.
    pub fn get_keys(&self) -> AllocResult<Term> {
        let mut heap = self.heap.lock();
        let dictionary = self.dictionary.lock();

        let len = dictionary.len();
        let need_in_words = Cons::need_in_words_from_len(len);

        if need_in_words <= heap.heap_available() {
            let entry_vec: Vec<Term> = dictionary.keys().copied().collect();

            Ok(heap
                .list_from_slice(&entry_vec)
                .map(|list| list.into())
                .unwrap())
        } else {
            Err(alloc!())
        }
    }

    /// Returns list of all keys from the process dictionary that have `value`.
    pub fn get_keys_from_value(&self, value: Term) -> AllocResult<Term> {
        let mut heap = self.heap.lock();
        let dictionary = self.dictionary.lock();

        let key_vec: Vec<Term> = dictionary
            .iter()
            .filter_map(|(entry_key, entry_value)| {
                if entry_value == &value {
                    Some(*entry_key)
                } else {
                    None
                }
            })
            .collect();

        heap.list_from_slice(&key_vec).map(|list| list.into())
    }

    /// Removes all key/value pairs from process dictionary and returns list of the entries.
    pub fn erase_entries(&self) -> AllocResult<Term> {
        let mut heap = self.heap.lock();
        let mut dictionary = self.dictionary.lock();

        let len = dictionary.len();
        let entry_need = Tuple::layout_for_len(2);
        let entry_need_in_words = erts::to_word_size(entry_need.size());
        let need_in_words = Cons::need_in_words_from_len(len) + len * entry_need_in_words;

        if need_in_words <= heap.heap_available() {
            let entry_vec: Vec<Term> = dictionary
                .drain()
                .map(|(key, value)| heap.tuple_from_slice(&[key, value]).unwrap().into())
                .collect();

            Ok(heap
                .list_from_slice(&entry_vec)
                .map(|list| list.into())
                .unwrap())
        } else {
            Err(alloc!())
        }
    }

    /// Removes key/value pair from process dictionary and returns value for `key`.  If `key` is not
    /// there, it returns `:undefined`.
    pub fn erase_value_from_key(&self, key: Term) -> Term {
        assert!(key.is_valid(), "invalid key term for process dictionary");

        match self.dictionary.lock().remove(&key) {
            None => atom!("undefined"),
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

    /// Inserts roots from the process into the given root set.
    /// This includes all process dictionary entries.
    #[inline]
    pub fn base_root_set(&self, rootset: &mut RootSet) {
        for (k, v) in self.dictionary.lock().iter() {
            rootset.push(k as *const _ as *mut _);
            rootset.push(v as *const _ as *mut _);
        }
    }

    /// Performs a garbage collection, using the provided root set
    ///
    /// The result is either `Ok(reductions)`, where `reductions` is the estimated cost
    /// of the collection in terms of reductions; or `Err(GcError)` containing the reason
    /// why garbage collection failed. Typically this will be due to attempting a minor
    /// collection and discovering that a full sweep is needed, rather than doing so automatically,
    /// the decision is left up to the caller to make. Other errors are described in the
    /// `GcError` documentation.
    ///
    /// `need` is specified in words.
    #[inline]
    pub fn garbage_collect(&self, need: usize, roots: &mut [Term]) -> Result<usize, GcError> {
        let mut heap = self.heap.lock();
        // The roots passed in here are pointers to the native stack/registers, all other roots
        // we are able to pick up from the current process context
        let mut rootset = RootSet::new(roots);
        self.base_root_set(&mut rootset);
        // Initialize the collector with the given root set
        heap.garbage_collect(self, need, rootset)
    }

    /// Cleans up any linked HeapFragments which should have had any live
    /// references moved out by the time this is called.
    ///
    /// Since no live data is contained in these fragments, we can simply
    /// walk the list of fragments and drop them in place, freeing the memory
    /// associated with them. It is expected that heap fragments have a `Drop` impl
    /// that handles this last step.
    fn sweep_off_heap(&self) {
        // When we drop the `HeapFragment`, its `Drop` implementation executes the
        // destructor of the term stored in the fragment, and then frees the memory
        // backing it. Since this takes care of any potential clean up we may need
        // to do automatically, we don't have to do any more than that here, at least
        // for now. In the future we may need to have more control over this, but
        // not in the current state of the system
        let mut off_heap = self.off_heap.lock();
        let mut cursor = off_heap.front_mut();
        while let Some(fragment_ref) = cursor.remove() {
            let fragment_ptr = UnsafeRef::into_raw(fragment_ref);
            unsafe { ptr::drop_in_place(fragment_ptr) };
        }
    }

    /// Determines if we should try and grow the heap even when not necessary
    #[inline]
    pub(super) fn should_force_heap_growth(&self) -> bool {
        self.flags.are_set(ProcessFlags::GrowHeap)
    }

    /// Returns true if the given pointer belongs to memory owned by this process
    #[inline]
    pub fn is_owner<T>(&self, ptr: *const T) -> bool
    where
        T: ?Sized,
    {
        let heap = self.heap.lock();
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
    pub fn run(arc_process: &Arc<Process>) -> code::Result {
        arc_process.start_running();

        // `code` is expected to set `code` before it returns to be the next spot to continue
        let option_code = arc_process
            .code_stack
            .lock()
            .get(0)
            .map(|frame| frame.code());

        let code_result = match option_code {
            Some(code) => code(arc_process),
            None => Ok(arc_process.exit_normal(anyhow!("Out of code").into())),
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

    pub fn exit(&self, reason: Term, source: ArcError) {
        self.reduce();
        self.exception(exit!(reason, source));
    }

    pub fn exit_normal(&self, source: ArcError) {
        self.exit(atom!("normal"), source);
    }

    pub fn is_exiting(&self) -> bool {
        if let Status::Exiting(_) = *self.status.read() {
            true
        } else {
            false
        }
    }

    pub fn exception(&self, exception: RuntimeException) {
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
    pub fn call_code(arc_process: &Arc<Process>) -> code::Result {
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

    pub fn current_definition(&self) -> Option<Definition> {
        self.code_stack
            .lock()
            .get(0)
            .map(|frame| frame.definition().clone())
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

    pub fn remove_last_frame(&self, stack_used: usize) {
        self.stack_popn(stack_used);
        let mut locked_code_stack = self.code_stack.lock();

        // the last frame is at the 2nd spot because the true last frame is `lumen:out_of_code/1`.
        assert_eq!(locked_code_stack.len(), 2);

        // unwrap to ensure there is a frame to replace
        locked_code_stack.pop().unwrap();
    }

    pub fn return_from_call(&self, stack_used: usize, term: Term) -> AllocResult<()> {
        let mut locked_heap = self.heap.lock();
        // ensure there is space on the stack for return value before messing with the stack and
        // breaking reentrancy
        if locked_heap.stack_available() + stack_used > 0 {
            if stack_used > 0 {
                // Heap cannot pop n 0
                locked_heap.stack_popn(stack_used);
            }

            let mut locked_stack = self.code_stack.lock();

            // remove current frame.  The caller becomes the top frame, so it's
            // `module_function_arity` will be returned from
            // `current_module_function_arity`.
            locked_stack.pop();

            // no caller, return value is thrown away, process will exit when
            // `Scheduler.run_once` detects it has no frames.
            if locked_stack.len() > 0 {
                unsafe {
                    let ptr = locked_heap.alloca(1).unwrap().as_ptr();
                    ptr::write(ptr, term);
                }
            }

            Ok(())
        } else {
            Err(alloc!())
        }
    }

    pub fn stacktrace(&self) -> stack::Trace {
        self.code_stack.lock().trace()
    }
}

#[cfg(test)]
impl Process {
    #[inline]
    fn young_heap_used(&self) -> usize {
        use crate::erts::process::alloc::GenerationalHeap;

        let heap = self.heap.lock();
        heap.heap().young_generation().heap_used()
    }

    #[inline]
    fn old_heap_used(&self) -> usize {
        use crate::erts::process::alloc::GenerationalHeap;

        let heap = self.heap.lock();
        heap.heap().old_generation().heap_used()
    }

    #[inline]
    fn has_old_heap(&self) -> bool {
        use crate::erts::process::alloc::GenerationalHeap;

        let heap = self.heap.lock();
        heap.heap().old_generation().active()
    }
}

impl fmt::Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.pid)?;

        match *self.registered_name.read() {
            Some(registered_name) => write!(f, " ({:?})", registered_name),
            None => Ok(()),
        }
    }
}

impl fmt::Display for Process {
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

impl Eq for Process {}

impl Hash for Process {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pid.hash(state);
    }
}

impl PartialEq for Process {
    fn eq(&self, other: &Self) -> bool {
        self.pid == other.pid
    }
}

unsafe impl Send for Process {}
unsafe impl Sync for Process {}

type Reductions = u16;

// [BEAM statuses](https://github.com/erlang/otp/blob/551d03fe8232a66daf1c9a106194aa38ef660ef6/erts/emulator/beam/erl_process.c#L8944-L8972)
#[derive(Debug, PartialEq)]
pub enum Status {
    Runnable,
    Running,
    Waiting,
    Exiting(RuntimeException),
}

impl Default for Status {
    fn default() -> Status {
        Status::Runnable
    }
}

#[cfg(test)]
mod test;
