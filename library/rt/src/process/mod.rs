mod flags;
mod generator;
mod heap;
mod id;
pub mod link;
pub mod monitor;
pub mod signals;
mod spawn;
mod stack;
mod system_tasks;

use alloc::alloc::{AllocError, Allocator, Layout};
use alloc::boxed::Box;
use alloc::fmt;
use alloc::sync::{Arc, Weak};
use core::assert_matches::assert_matches;
use core::cell::UnsafeCell;
use core::cmp;
use core::convert::AsRef;
use core::mem;
use core::num::{NonZeroU64, NonZeroUsize};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use firefly_alloc::fragment::HeapFragmentList;
use firefly_alloc::heap::Heap;
use firefly_system::sync::{Atomic, Mutex, MutexGuard};

use crossbeam::deque::Injector;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListAtomicLink};

use log::trace;

use crate::error::{ErlangException, ErrorCode, ExceptionClass, ExceptionFlags, ExceptionInfo};
use crate::function::ModuleFunctionArity;
use crate::gc::{GcError, RootSet, SemispaceProcessHeap};
use crate::scheduler::SchedulerId;
use crate::services::registry::WeakAddress;
use crate::term::{
    atoms, Atom, LayoutBuilder, OpaqueTerm, Pid, ReferenceId, Term, TermFragment, Tuple,
};

pub use self::flags::{MaxHeapSize, Priority, ProcessFlags, StatusFlags};
pub use self::generator::{Continuation, ContinuationResult, Generator, GeneratorState};
pub use self::heap::ProcessHeap;
pub use self::id::{ProcessId, ProcessIdError};
pub use self::spawn::*;
pub use self::stack::{ProcessStack, Register, StackFrame, ARG0_REG, CP_REG, RETURN_REG};
pub use self::system_tasks::{SystemTask, SystemTaskType};

use self::link::LinkTree;
use self::monitor::{MonitorList, MonitorTree};
use self::signals::{FlushType, Message, SendResult, Signal, SignalEntry, SignalQueue};
use self::system_tasks::SystemTaskList;

/// A convenient type alias for the intrusive linked list type which is used by schedulers
pub type ProcessList = LinkedList<ProcessAdapter>;

intrusive_adapter!(pub ProcessAdapter = Arc<Process>: Process { link: LinkedListAtomicLink });

/// This represents the state of the process timer at any given time.
///
/// The process timer is used when a process is suspended while waiting
/// on an operation to complete or for a signal to be received. Since the
/// timer and the process may be executing on different threads, this enumeration
/// allows us to determine when certain events have occured from both threads and
/// correctly handle them.
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProcessTimer {
    /// There is no timer set for this process
    #[default]
    None,
    /// There was a timer set for this process, but it has timed out
    TimedOut,
    /// There is an active timer set for this process
    Active(*const ()),
}

impl firefly_system::sync::Atom for ProcessTimer {
    type Repr = usize;

    #[inline]
    fn pack(self) -> Self::Repr {
        match self {
            Self::None => 0,
            Self::TimedOut => 1,
            Self::Active(ptr) => ptr as usize,
        }
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        match raw {
            0 => Self::None,
            1 => Self::TimedOut,
            n => Self::Active(n as *const ()),
        }
    }
}

#[derive(Copy, Clone)]
pub enum ContinueExitPhase {
    Timers = 0,
    UsingDb,
    CleanSysTasks,
    Free,
    CleanSysTasksAfter,
    Links,
    Monitors,
    HandleProcessSignals,
    DistSend,
    DistLinks,
    DistMonitors,
    PendingSpawnMonitors,
    Done,
}

/// This struct contains all of the fields owned by the process scheduler
///
/// Interacting with this data requires obtaining a [`ProcessLock`].
///
/// NOTE: There are a couple of fields implicitly managed by the process scheduler,
/// which are not members of this struct, as they do not require the main lock to read,
/// only to write.
pub struct SchedulerData {
    /// The offset or pointer to the instruction at which this process is currently executing.
    pub ip: usize,
    /// The reduction counter for this process
    pub reductions: usize,
    /// The number of bytes needed when the next garbage collection is performed
    ///
    /// If zero, no requirement is imposed on the collector. If non-zero, the collector
    /// must free/acquire at least that much memory, or an error is raised.
    pub gc_needed: usize,
    /// The percentage of used to unused space at which a collection is triggered
    pub gc_threshold: f64,
    /// The number of minor collections that have occurred since the last full sweep
    pub gc_count: usize,
    /// The number of schedules remaining for a low priority process
    pub schedule_count: u8,
    /// A unique number counter for this process
    pub uniq: NonZeroU64,
    /// The set of internal process flags which are controlled by the scheduler
    pub flags: ProcessFlags,
    /// If `timer` is set, this field holds the reference for the timer
    pub timer_ref: ReferenceId,
    /// Used to reschedule the process when suspended
    pub injector: Arc<Injector<Arc<Process>>>,
    /// Used to handle yielding BIFs/NIFs
    pub awaiting: Option<Generator>,
    /// Stores the target of the trap instruction
    pub trap: Option<firefly_bytecode::FunId>,
    /// This field represents metadata about the current exception and how it should be handled.
    ///
    /// For exceptions which have already been allocated, the `current_exception` field holds
    /// the data for it. This field contains flags which control the behavior of exception handling
    /// for the current exception, and in some cases is used without a `current_exception`.
    ///
    /// `ExceptionInfo::is_empty()` returns true when there is no active exception
    pub exception_info: ExceptionInfo,
    /// Used during termination of a process
    pub continue_exit: ContinueExitPhase,
    /// The stack for this process
    ///
    /// This is only ever accessed by the process itself while executing.
    ///
    /// Conceptually the stack is a vector of terms. Call frames are potentially overlapping
    /// windows into this stack, where each frame gets up to 256 addressible stack slots, or
    /// "registers", which allow instructions to directly reference a local on the stack in
    /// that frame.
    ///
    /// Each call which conceptually grows the stack will reserve at least as many stack slots
    /// as are required by the callee, which may trigger the underlying stack memory to be
    /// reallocated.
    ///
    /// The stack is the primary source of roots for garbage collection, in addition to
    /// `initial_arguments`, and the process dictionary.
    pub stack: ProcessStack,
    /// The heap for this process
    pub heap: SemispaceProcessHeap,
    /// Used to track monitors from local sources
    pub monitored_by: MonitorList,
    /// Used to track monitored targets
    pub monitored: MonitorTree,
    /// Used to track links
    pub links: LinkTree,
    /// The group leader of the current process.
    ///
    /// This will only ever be `None` for the init process
    group_leader: Option<Pid>,
    /// The heap fragment list for this process
    pub heap_fragments: HeapFragmentList,
    /// The system task queues, one for each priority: low, normal, high, max
    pub system_tasks: [SystemTaskList; 4],
}
impl SchedulerData {
    pub fn set_exception_info(&mut self, exception: Box<ErlangException>) {
        assert!(exception.fragment.is_none());
        self.exception_info.flags = match exception.kind {
            ExceptionClass::Error => ExceptionFlags::ERROR,
            ExceptionClass::Exit => ExceptionFlags::EXIT,
            ExceptionClass::Throw => ExceptionFlags::THROW,
        };
        self.exception_info.reason = match exception.reason() {
            Term::Atom(a) => a.into(),
            Term::Tuple(tuple) => match tuple[0].into() {
                Term::Atom(a) => a.into(),
                _ => ErrorCode::Other(exception.kind.into()),
            },
            _ => ErrorCode::Other(exception.kind.into()),
        };
        self.exception_info.value = exception.reason;
        self.exception_info.trace = Some(exception.trace());
    }
}

/// This type is used to represent ownership of a `Process`, and the holder of
/// a `ProcessLock` has exclusive mutable access to the `SchedulerData` of the
/// protected `Process`. References to a `ProcessLock` are also used as a token
/// to indicate the right to modify certain fields that are otherwise free to
/// read using other synchronization primitives (e.g. atomics) or directly in
/// the case of certain static/read-only fields.
///
/// This type provides APIs for accessing fields of `Process` which are not directly
/// protected by the lock, but which are implicitly owned by the holder of a `ProcessLock`.
/// These fields cannot be otherwise read.
pub struct ProcessLock<'a> {
    process: &'a Process,
    guard: MutexGuard<'a, SchedulerData>,
}
impl<'a> ProcessLock<'a> {
    fn new(process: &'a Process) -> Self {
        let guard = process.scheduler_data.lock();
        Self { process, guard }
    }

    /// Get a new strong `Arc` reference to the locked process
    pub fn strong(&self) -> Arc<Process> {
        unsafe {
            Arc::increment_strong_count(self.process);
            Arc::from_raw(self.process)
        }
    }

    /// Get a new `Weak` reference to the locked process
    pub fn weak(&self) -> Weak<Process> {
        let process = self.strong();
        Arc::downgrade(&process)
    }
}
impl<'a> Deref for ProcessLock<'a> {
    type Target = SchedulerData;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}
impl<'a> DerefMut for ProcessLock<'a> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}
impl<'a> AsRef<Process> for ProcessLock<'a> {
    #[inline]
    fn as_ref(&self) -> &Process {
        self.process
    }
}

/// This is the process structure containing all of the metadata about a process.
///
/// A process has conceptually three levels of access:
///
/// 1. Read-only or thread-safe access to a limited set of fields by any thread with a reference
/// 2. Mutable access to the private process signal queue by holding the signal queue lock
/// 3. Mutable access to the process scheduler data by holding the main lock
///
/// Number 3 does not imply 2, though it may often be the case that the holder of
/// the main lock also holds the signal queue lock. It may also be the case that certain read-only
/// fields which are publically accessible can be changed only by the holder of one of these two
/// locks. All such fields are safe for concurrent access however.
// These targets all have 128-byte cache-line size
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
// These targets all have 32-byte cache-line size
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64"
    ),
    repr(align(32))
)]
// x86 and Wasm have 64-byte cache-line size, and we assume 64 bytes for all other platforms
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "riscv64"
    )),
    repr(align(64))
)]
pub struct Process {
    /// The intrusive list link reserved for use by the owning scheduler of this process
    ///
    /// This link must not be modified by anyone other than the owning scheduler.
    ///
    /// Attempting to modify the link inappropriately will result in a panic. This is because
    /// the link does not allow modification while it is active, triggering a panic when an attempt
    /// to do so occurs. It is not guaranteed that this panic will happen in the non-owning
    /// context, as schedulers _do_ temporarily remove processes from their queues during
    /// execution of those processes, and then re-insert them in the queue after - should the
    /// link have been stolen in the interim, the owning scheduler may be the one to panic.
    /// Needless to say, incorrect usage will not go unnoticed.
    ///
    /// This field is the only scheduler-owned field which is not protected by the main lock
    link: LinkedListAtomicLink,
    /// This field provides the current, owning scheduler identifier
    ///
    /// It is not permitted to modify this field without holding the main process lock, but it
    /// is always safe to read.
    scheduler_id: Atomic<SchedulerId>,
    /// This field represents the main process lock.
    ///
    /// The data it contains is owned by, and must only by modified by, a specific scheduler
    /// instance.
    scheduler_data: Mutex<SchedulerData>,
    /// The unique process id of the current process
    ///
    /// This is set on creation and never changes, so is safe to be read concurrently
    id: ProcessId,
    /// The parent of the current process.
    ///
    /// If None, then this process has no parent, which is only ever true of the init process
    ///
    /// This is set on creation and never changes, so is safe to be read concurrently
    parent: Option<Pid>,

    /// The registered name of the current process
    ///
    /// If this process has a registered name, the pointer is non-null and points to the atom
    /// representing the name of this process. Otherwise, the pointer is null.
    registered_name: Atomic<Atom>,
    /// The module/function/arity of the start function for this process
    ///
    /// This is set on creation and never changes, so is safe to be read concurrently
    ///
    /// NOTE: We may not need to keep this field, but instead store this information in the process
    /// dictionary under the '$initial_call' key. It is not used once the process has been
    /// scheduled the first time.
    pub initial_call: ModuleFunctionArity,
    /// Stores the initial arguments of this process
    ///
    /// None means there were no initial arguments, or they have been discarded
    ///
    /// This is set on creation, and then set to None the first time to the process is scheduled.
    /// It is never accessed by any other thread.
    initial_arguments: UnsafeCell<Option<Term>>,
    /// This field contains the current process timer state
    ///
    /// A process timer is set when an operation should block until some amount of time
    /// has passed. The timer may be unset/empty, canceled or time out, so we need a special
    /// enumeration to represent this rather than a simple `Option<NonNull<Timer>>`.
    pub timer: Atomic<ProcessTimer>,
    /// The current process status and related flags
    ///
    /// This is a bitset which allows modification from multiple threads, but not all flags
    /// are allowed to be modified by anyone. Instead, the process (or its owning scheduler)
    /// owns status and flags which represent what the process is currently doing. Certain
    /// interactions from other processes/ports may modify a limited set of flags having to
    /// do with the signal queue, but this is internal to the runtime system.
    ///
    /// There are a few fields following this one which contain other process flags not tracked
    /// in this bitset, which are read-only by anyone, but may only be modified by the process.
    pub status: Atomic<StatusFlags>,
    pub error_handler: Atomic<Atom>,
    pub fullsweep_after: AtomicUsize,
    pub min_heap_size: Option<NonZeroUsize>,
    pub min_bin_vheap_size: Option<NonZeroUsize>,
    pub max_heap_size: Atomic<MaxHeapSize>,
    /// The mailbox/signal queue for this process
    ///
    /// The signal queue is a thread-safe structure which internally maintains multiple queues for
    /// in-transit and recieved signals (including messages) for this process. It handles
    /// prioritization automatically, and provides support for stateful receives (i.e. cursor
    /// into the queue, etc).
    ///
    /// Signals can be enqueued by any process at any time, which will place them in-transit. From
    /// there, the receiving process moves them into its internal queue and begins handling
    /// them one-by-one.
    pub signals: SignalQueue,
}
impl fmt::Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let scheduler_id = self.scheduler_id.load(Ordering::Relaxed);
        let status = self.status(Ordering::Relaxed);
        f.debug_struct("Process")
            .field("link", &self.link)
            .field("scheduler_id", &scheduler_id)
            .field("id", &self.id)
            .field("parent", &self.parent)
            .field("initial_call", &self.initial_call)
            .field("status", &status)
            .finish()
    }
}

/// Processes can be scheduled across threads
unsafe impl Send for Process {}

/// Processes can be safely written to from multiple threads
///
/// The invariants for this property are documented for each field, as well as
/// on each function which may read/write a field that might be accessed from
/// another thread.
unsafe impl Sync for Process {}

impl Process {
    pub const MAX_REDUCTIONS: usize = 4000;

    pub fn new(
        scheduler_id: SchedulerId,
        parent: Option<Pid>,
        group_leader: Option<Pid>,
        initial_call: ModuleFunctionArity,
        initial_arguments: &[OpaqueTerm],
        injector: Arc<Injector<Arc<Process>>>,
        opts: SpawnOpts,
    ) -> Arc<Self> {
        let id = ProcessId::next();

        // Make sure the heap is at least large enough to hold `initial_arguments`
        let min_heap_size = cmp::max(
            ProcessHeap::DEFAULT_SIZE,
            opts.min_heap_size
                .map(|sz| sz.get())
                .unwrap_or(ProcessHeap::DEFAULT_SIZE),
        );
        let heap = if initial_arguments.is_empty() {
            SemispaceProcessHeap::new(ProcessHeap::new(min_heap_size), ProcessHeap::empty())
        } else {
            let mut lb = LayoutBuilder::new();
            for arg in initial_arguments.iter().copied() {
                if arg.is_box() {
                    let t: Term = arg.into();
                    lb += t.layout();
                }
            }
            lb.build_tuple(initial_arguments.len());
            let layout = lb.finish();
            let required_heap_size = ProcessHeap::next_size(layout.size());
            let heap_size = cmp::max(required_heap_size, min_heap_size);
            SemispaceProcessHeap::new(ProcessHeap::new(heap_size), ProcessHeap::empty())
        };

        // Initialize the stack and get ready to write the initial arguments
        let mut stack = ProcessStack::default();
        unsafe {
            // Default return value
            stack.store(0, OpaqueTerm::NIL);
            // Default return address is a null pointer value
            stack.store(1, OpaqueTerm::NONE);
            // Arguments to initial call
            stack.alloca(initial_arguments.len());
        }

        // Move initial arguments onto process heap
        let initial_arguments = if initial_arguments.is_empty() {
            None
        } else {
            let mut args = Tuple::from_slice(initial_arguments, &heap).unwrap();
            {
                for (i, arg) in initial_arguments.iter().copied().enumerate() {
                    if !arg.is_box() {
                        continue;
                    }
                    let term: Term = arg.into();
                    let term = term.clone_to_heap(&heap).unwrap();
                    let term = term.into();
                    args[i] = term;
                    // Store each argument in it's corresponding register on the stack
                    stack.store((i + stack::RESERVED_REGISTERS) as Register, term);
                }
            }
            Some(Term::Tuple(args))
        };

        Arc::new(Self {
            link: LinkedListAtomicLink::new(),
            scheduler_data: Mutex::new(SchedulerData {
                ip: 0,
                reductions: 0,
                gc_needed: 0,
                gc_threshold: 0.75,
                gc_count: 0,
                schedule_count: 0,
                uniq: unsafe { NonZeroU64::new_unchecked(1) },
                timer_ref: ReferenceId::zero(),
                injector,
                awaiting: None,
                trap: None,
                flags: ProcessFlags::empty(),
                exception_info: ExceptionInfo::default(),
                continue_exit: ContinueExitPhase::Timers,
                stack,
                heap,
                monitored_by: Default::default(),
                monitored: Default::default(),
                links: Default::default(),
                group_leader,
                heap_fragments: HeapFragmentList::default(),
                system_tasks: [
                    SystemTaskList::default(),
                    SystemTaskList::default(),
                    SystemTaskList::default(),
                    SystemTaskList::default(),
                ],
            }),
            scheduler_id: Atomic::new(scheduler_id),
            parent,
            id,
            registered_name: Atomic::new(atoms::Undefined),
            initial_call,
            initial_arguments: UnsafeCell::new(initial_arguments),
            timer: Atomic::new(Default::default()),
            status: Atomic::new(StatusFlags::default() | StatusFlags::ACTIVE | opts.priority),
            error_handler: Atomic::new(atoms::Undefined),
            fullsweep_after: AtomicUsize::new(opts.fullsweep_after.unwrap_or(usize::MAX)),
            min_heap_size: opts.min_heap_size,
            min_bin_vheap_size: opts.min_bin_vheap_size,
            max_heap_size: Atomic::new(opts.max_heap_size),
            signals: SignalQueue::default(),
        })
    }

    /// Acquires the main process lock for this process
    #[inline(always)]
    pub fn lock<'a>(&'a self) -> ProcessLock<'a> {
        ProcessLock::new(self)
    }

    /// Sets the initial instruction pointer for a new process
    pub fn set_instruction_pointer(&mut self, ip: usize) {
        self.scheduler_data.get_mut().ip = ip;
    }

    pub fn parent(&self) -> Option<Pid> {
        self.parent.clone()
    }

    /// Returns true if this process is the init process (i.e. the first spawned process)
    pub fn is_init(&self) -> bool {
        unsafe { self.id == ProcessId::from_raw(1 << 32) }
    }

    /// Returns the `ProcessId` for this process
    #[inline]
    pub fn id(&self) -> ProcessId {
        self.id
    }

    /// Returns a `Pid` corresponding to this process
    #[inline]
    pub const fn pid(&self) -> Pid {
        Pid::new_local(self.id)
    }

    /// Returns a `WeakAddress` corresponding to this process
    #[inline]
    pub const fn addr(&self) -> WeakAddress {
        WeakAddress::Process(self.pid())
    }

    /// Returns the `SchedulerId` currently responsible for this process
    #[inline]
    pub fn scheduler_id(&self) -> SchedulerId {
        self.scheduler_id.load(Ordering::Acquire)
    }

    /// Get a reference to the signal queue for this process
    #[inline]
    pub fn signals(&self) -> &SignalQueue {
        &self.signals
    }

    /// Returns the registered name of this process, if one exists
    pub fn registered_name(&self) -> Option<Atom> {
        let name = self.registered_name.load(Ordering::Acquire);
        if name == atoms::Undefined {
            None
        } else {
            Some(name)
        }
    }

    /// Sets the registered name of this process
    ///
    /// This function returns `Ok` if the process was unregistered when this function was called,
    /// otherwise it returns `Err` with the previously registered name of this process. This
    /// function will never replace an already registered name.
    pub fn register_name(&self, name: Atom) -> Result<(), Atom> {
        assert_ne!(
            name,
            atoms::Undefined,
            "undefined is not a valid registered name"
        );
        match self.registered_name.compare_exchange(
            atoms::Undefined,
            name,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            Ok(_) => Ok(()),
            Err(existing) => {
                // Already registered
                Err(existing)
            }
        }
    }

    /// Removes the registered name of this process
    ///
    /// This function returns `Ok` if the process was registered when this function was called,
    /// otherwise it returns `Err` which implies that the process had no registered name already.
    pub fn unregister_name(&self) -> Result<(), ()> {
        let prev = self
            .registered_name
            .swap(atoms::Undefined, Ordering::Release);
        if prev == atoms::Undefined {
            Err(())
        } else {
            Ok(())
        }
    }

    /// Returns the max heap size configuration for this process
    pub fn max_heap_size(&self) -> MaxHeapSize {
        self.max_heap_size.load(Ordering::Relaxed)
    }

    /// Reads the current process status flags with the given memory ordering
    ///
    /// Any read which needs a happens-before relationship with another write should use `Acquire`,
    /// and the write should use `Release` ordering. For everything else, `Relaxed` is a better
    /// choice.
    #[inline]
    pub fn status(&self, ordering: Ordering) -> StatusFlags {
        self.status.load(ordering)
    }

    /// Enables all of the status flags in `flags`, using the given memory ordering for the write.
    ///
    /// Returns the previous status flags
    ///
    /// Any uses of this function which are paired with a read that needs consistent ordering
    /// should use `Release`. For all other uses, `Relaxed` is a better choice.
    pub fn set_status_flags(&self, flags: StatusFlags, ordering: Ordering) -> StatusFlags {
        self.status.fetch_or(flags, ordering)
    }

    /// Performs a compare-exchange on the current flag set, setting them to `new` if the current
    /// flags match `current`.
    ///
    /// Returns a result containing the current flags, where `Ok` indicates success, `Err` indicates
    /// that the current flags were different than expected.
    pub fn cmpxchg_status_flags(
        &self,
        current: StatusFlags,
        new: StatusFlags,
    ) -> Result<StatusFlags, StatusFlags> {
        self.status
            .compare_exchange(current, new, Ordering::Release, Ordering::Acquire)
    }

    /// Disables all of the status flags in `flags`, using the given memory ordering for the write.
    ///
    /// Returns the previous status flags
    ///
    /// Any uses of this function which are paired with a read that needs consistent ordering
    /// should use `Release`. For all other uses, `Relaxed` is a better choice.
    pub fn remove_status_flags(&self, flags: StatusFlags, ordering: Ordering) -> StatusFlags {
        self.status.fetch_and(!flags, ordering)
    }

    /// Get the current process timer state
    #[inline]
    pub fn timer(&self) -> ProcessTimer {
        self.timer.load(Ordering::Acquire)
    }

    /// Sets the process timer to `ProcessTimer::TimedOut`, if the timer is `current`
    ///
    /// This has the following effects on process status/flags:
    ///
    /// * Removes `IN_TIMER_QUEUE` process flag
    /// * Sets the `TIMEOUT` process flag.
    /// * Sets the `ACTIVE` status flag
    ///
    /// The process is rescheduled via its owning scheduler as well
    #[inline]
    pub fn set_timeout(self: Arc<Self>, current: ProcessTimer) -> Result<(), ProcessTimer> {
        // We must acquire the main lock to proceed
        let process = self.clone();
        let mut scheduler_data = self.scheduler_data.lock();
        self.timer.compare_exchange(
            current,
            ProcessTimer::TimedOut,
            Ordering::Release,
            Ordering::SeqCst,
        )?;
        // Update the process flags to indicate that the process is no longer
        // in the timer queue, and has timed out on a blocking operation
        scheduler_data.flags &= !ProcessFlags::IN_TIMER_QUEUE;
        scheduler_data.flags |= ProcessFlags::TIMEOUT;

        // If the status is currently suspended, mark it ready for scheduling
        let status = self.status.fetch_or(StatusFlags::ACTIVE, Ordering::Release);
        if status.contains(StatusFlags::SUSPENDED) {
            self.status
                .fetch_and(!StatusFlags::SUSPENDED, Ordering::Release);
            // We hold the process lock, so it is not possible for this process to
            // be currently executing on a scheduler. Furthermore we just confirmed
            // that the process is suspended, which together means that it can't be
            // in any scheduler run queues already.
            trace!(target: "process", "waking up process for timeout");
            scheduler_data.injector.push(process);
        }
        Ok(())
    }

    /// Send `message` from `sender` to this process
    pub fn send(self: Arc<Self>, sender: WeakAddress, message: Term) -> Result<(), ()> {
        let fragment = TermFragment::new(message).unwrap();
        self.send_fragment(sender, fragment)
    }

    /// Send a message from `sender` and allocated in `fragment`, to this process
    pub fn send_fragment(
        self: Arc<Self>,
        sender: WeakAddress,
        fragment: TermFragment,
    ) -> Result<(), ()> {
        self.do_send_message(
            SignalEntry::new(Signal::Message(Message {
                sender,
                message: fragment,
            })),
            false,
        )
    }

    /// Sends a raw signal entry to this process
    #[inline]
    pub fn send_signal(self: Arc<Self>, entry: Box<SignalEntry>) -> Result<(), ()> {
        if entry.is_message() {
            self.do_send_message(entry, false)
        } else {
            self.do_send_signal(entry, false)
        }
    }

    /// Sends a raw signal entry to this process, flushing the in-transit queue first
    #[inline]
    pub fn send_signal_after_flush(self: Arc<Self>, entry: Box<SignalEntry>) -> Result<(), ()> {
        if entry.is_message() {
            self.do_send_message(entry, true)
        } else {
            self.do_send_signal(entry, true)
        }
    }

    fn do_send_message(
        self: Arc<Self>,
        entry: Box<SignalEntry>,
        force_flush: bool,
    ) -> Result<(), ()> {
        let mut status = self.status(Ordering::Relaxed);
        trace!(target: "process", "sending message to process with status {:?}, force_flush={}", status, force_flush);
        if status.contains(StatusFlags::EXITING) {
            trace!(target: "process", "process is exiting, dropping message");
            return Err(());
        }

        if force_flush {
            let mut queue = self.signals.lock();
            // Acquire the status again since we may have suspended waiting on the lock
            status = self.status.load(Ordering::Relaxed);

            if status.contains(StatusFlags::FREE) {
                return Err(());
            }

            queue.push_private(entry);
        } else {
            self.signals.push(entry);
        }

        // Acquire the status again since we may have context-switched on a lock when pushing
        status = self.status(Ordering::Relaxed);

        // If the process is currently active, we're done
        if status.intersects(StatusFlags::RUNNING | StatusFlags::ACTIVE) {
            trace!(target: "process", "recipient is currently active, send considered successful");
            return Ok(());
        }

        // Finally, mark the process active/non-suspended if it wasn't already
        trace!(target: "process", "recipient is currently inactive, attempting to reschedule it");
        loop {
            if status.contains(StatusFlags::EXITING | StatusFlags::ACTIVE) {
                return Ok(());
            }
            let mut new_status = status | StatusFlags::ACTIVE;
            new_status.remove(StatusFlags::SUSPENDED);
            match self.cmpxchg_status_flags(status, new_status) {
                Ok(prev) => {
                    // If we removed the suspended flag, then we're in the clear
                    // to reschedule this process as long as we can acquire the
                    // process lock
                    if prev.contains(StatusFlags::SUSPENDED) {
                        let process = self.clone();
                        if let Some(guard) = self.scheduler_data.try_lock() {
                            trace!(target: "process", "receipient has been woken up");
                            guard.injector.push(process);
                            break;
                        }
                    }
                    trace!(target: "process", "receipient has been woken up by another process");
                    break;
                }
                Err(current) => {
                    status = current;
                }
            }
        }

        Ok(())
    }

    fn do_send_signal(
        self: Arc<Self>,
        entry: Box<SignalEntry>,
        force_flush: bool,
    ) -> Result<(), ()> {
        let sender = entry.sender().unwrap_or(WeakAddress::System);
        let mut status;
        if sender.eq(self.as_ref()) {
            status = self.set_status_flags(StatusFlags::MAYBE_SELF_SIGNALS, Ordering::Relaxed);
            status |= StatusFlags::MAYBE_SELF_SIGNALS;
        } else {
            status = self.status(Ordering::Relaxed);
        }

        let is_process_info = entry.signal.is_process_info();
        if !force_flush && !is_process_info {
            let result = self.signals.push(entry);
            if result.contains(SendResult::NOTIFY_IN_TRANSIT) {
                if !status.contains(StatusFlags::HAS_IN_TRANSIT_SIGNALS) {
                    status = self
                        .status
                        .fetch_or(StatusFlags::HAS_IN_TRANSIT_SIGNALS, Ordering::Release);
                    status |= StatusFlags::HAS_IN_TRANSIT_SIGNALS;
                }
            }
            self.notify_new_signal(status);
            return Ok(());
        }

        {
            let mut queue = self.signals.lock();
            status = self.status.load(Ordering::Relaxed);

            if status.contains(StatusFlags::FREE) {
                return Err(());
            }

            let result = queue.push_private(entry);
            if result.contains(SendResult::NOTIFY_IN_TRANSIT) {
                status = self.status.fetch_or(
                    StatusFlags::HAS_PENDING_SIGNALS | StatusFlags::HAS_IN_TRANSIT_SIGNALS,
                    Ordering::Relaxed,
                );
            }
            // if unlikely(is_process_info) {
            //     check_push_msgq_len_offs_marker(rp, sig);
            // }
        }

        self.notify_new_signal(status);
        Ok(())
    }

    fn notify_new_signal(self: Arc<Self>, status: StatusFlags) {
        if !status.intersects(
            StatusFlags::EXITING | StatusFlags::ACTIVE_SYS | StatusFlags::HAS_IN_TRANSIT_SIGNALS,
        ) {
            self.active_sys_enqueue(status);
        }
    }

    fn active_sys_enqueue(self: Arc<Self>, mut status: StatusFlags) {
        loop {
            if status.contains(StatusFlags::FREE) {
                break;
            }
            match self.status.compare_exchange(
                status,
                status | StatusFlags::ACTIVE_SYS,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(prev) => {
                    // The process is currently suspended, we need to reschedule it
                    if prev.contains(StatusFlags::SUSPENDED) {
                        let process = self.clone();
                        if let Some(guard) = self.scheduler_data.try_lock() {
                            guard.injector.push(process);
                        }
                    }
                    break;
                }
                Err(current) => {
                    status = current;
                }
            }
        }
    }
}

impl<'a> ProcessLock<'a> {
    pub fn parent(&self) -> Option<Pid> {
        self.as_ref().parent()
    }

    #[inline]
    pub fn id(&self) -> ProcessId {
        self.as_ref().id
    }

    #[inline]
    pub fn pid(&self) -> Pid {
        self.as_ref().pid()
    }

    #[inline]
    pub fn next_unique(&mut self) -> NonZeroU64 {
        let id = self.guard.uniq;
        self.guard.uniq = id.checked_add(1).unwrap();
        id
    }

    #[inline]
    pub fn initial_call(&self) -> ModuleFunctionArity {
        self.as_ref().initial_call
    }

    #[inline]
    pub fn initial_arguments(&self) -> Option<Term> {
        let initial_args = unsafe { &*self.as_ref().initial_arguments.get() };
        initial_args.clone()
    }

    #[inline]
    pub fn addr(&self) -> WeakAddress {
        self.as_ref().addr()
    }

    #[inline]
    pub fn scheduler_id(&self) -> SchedulerId {
        self.as_ref().scheduler_id()
    }

    #[inline]
    pub fn reductions_left(&self) -> usize {
        Process::MAX_REDUCTIONS.saturating_sub(self.guard.reductions)
    }

    #[inline]
    pub fn signals(&self) -> &SignalQueue {
        self.as_ref().signals()
    }

    #[inline]
    pub fn registered_name(&self) -> Option<Atom> {
        self.as_ref().registered_name()
    }

    #[inline]
    pub fn register_name(&self, name: Atom) -> Result<(), Atom> {
        self.as_ref().register_name(name)
    }

    #[inline]
    pub fn unregister_name(&self) -> Result<(), ()> {
        self.as_ref().unregister_name()
    }

    #[inline]
    pub fn max_heap_size(&self) -> MaxHeapSize {
        self.as_ref().max_heap_size()
    }

    #[inline]
    pub fn status(&self, ordering: Ordering) -> StatusFlags {
        self.as_ref().status(ordering)
    }

    #[inline]
    pub fn set_status_flags(&self, flags: StatusFlags, ordering: Ordering) -> StatusFlags {
        self.as_ref().set_status_flags(flags, ordering)
    }

    #[inline]
    pub fn cmpxchg_status_flags(
        &self,
        current: StatusFlags,
        new: StatusFlags,
    ) -> Result<StatusFlags, StatusFlags> {
        self.as_ref().cmpxchg_status_flags(current, new)
    }

    #[inline]
    pub fn remove_status_flags(&self, flags: StatusFlags, ordering: Ordering) -> StatusFlags {
        self.as_ref().remove_status_flags(flags, ordering)
    }

    #[inline]
    pub fn timer(&self) -> ProcessTimer {
        self.as_ref().timer()
    }

    /// Send `message` from `sender` to this process
    pub fn send(&mut self, sender: WeakAddress, message: Term) -> Result<(), ()> {
        let fragment = TermFragment::new(message).unwrap();
        self.send_fragment(sender, fragment)
    }

    /// Send a message from `sender` and allocated in `fragment`, to this process
    pub fn send_fragment(&mut self, sender: WeakAddress, fragment: TermFragment) -> Result<(), ()> {
        self.do_send_message(
            SignalEntry::new(Signal::Message(Message {
                sender,
                message: fragment,
            })),
            false,
        )
    }

    /// Sends a raw signal entry to this process
    #[inline]
    pub fn send_signal(&mut self, entry: Box<SignalEntry>) -> Result<(), ()> {
        if entry.is_message() {
            self.do_send_message(entry, false)
        } else {
            self.strong().do_send_signal(entry, false)
        }
    }

    /// Sends a raw signal entry to this process, flushing the in-transit queue first
    #[inline]
    pub fn send_signal_after_flush(&mut self, entry: Box<SignalEntry>) -> Result<(), ()> {
        if entry.is_message() {
            self.do_send_message(entry, true)
        } else {
            self.strong().do_send_signal(entry, true)
        }
    }

    /// Performs the given type of flush on the signal queue
    pub fn flush_signals(&mut self, ty: FlushType) {
        use self::signals::SignalQueueFlags;

        assert!(!self
            .as_ref()
            .signals()
            .flags()
            .intersects(SignalQueueFlags::FLUSHING | SignalQueueFlags::FLUSHED));

        let force_flush;
        let enqueue;
        let fetch;
        match &ty {
            FlushType::Local => {
                force_flush = true;
                enqueue = false;
                fetch = true;
            }
            FlushType::Id(_) => {
                force_flush = false;
                enqueue = false;
                fetch = true;
            }
            FlushType::InTransit => {
                force_flush = false;
                enqueue = true;
                fetch = false;
            }
        }

        self.strong()
            .do_send_signal(Signal::flush(ty), force_flush)
            .expect("failed to send signal to ourselves");

        self.guard.flags |= ProcessFlags::DISABLE_GC;

        if fetch {
            let mut sigq = self.signals().lock();
            sigq.flush_buffers();
        }

        let sigq = self.signals();
        sigq.set_flags(SignalQueueFlags::FLUSHING);

        if enqueue {
            let sigq = sigq.lock();
            if !sigq.has_pending_signals() {
                sigq.set_flags(SignalQueueFlags::FLUSHED);
            }
        }
    }

    fn do_send_message(&mut self, entry: Box<SignalEntry>, force_flush: bool) -> Result<(), ()> {
        let mut status = self.status(Ordering::Relaxed);
        if status.contains(StatusFlags::EXITING) {
            return Err(());
        }

        if force_flush {
            let proc = self.as_ref();
            let mut queue = proc.signals.lock();
            // Acquire the status again since we may have suspended waiting on the lock
            status = proc.status.load(Ordering::Relaxed);

            if status.contains(StatusFlags::FREE) {
                return Err(());
            }

            queue.push_private(entry);
        } else {
            self.as_ref().signals.push(entry);
        }

        // Acquire the status again since we may have suspended on a lock when pushing
        status = self.status(Ordering::Relaxed);

        // If the process is currently active, we're done
        if status.intersects(StatusFlags::RUNNING | StatusFlags::ACTIVE) {
            return Ok(());
        }

        // Finally, set the ACTIVE status flag, so that the scheduler reschedules this process
        loop {
            if status.contains(StatusFlags::EXITING | StatusFlags::ACTIVE) {
                return Ok(());
            }
            match self.cmpxchg_status_flags(status, status | StatusFlags::ACTIVE) {
                Ok(_) => break,
                Err(current) => {
                    status = current;
                }
            }
        }

        // It is unexpected for us to be sending a message while holding the process lock
        // and simultaneously being suspended - for now we want to abort when that happens
        // so we can better understand the context.
        assert!(!status.contains(StatusFlags::SUSPENDED));

        Ok(())
    }

    #[inline]
    pub fn set_scheduler(&self, id: SchedulerId) {
        self.as_ref().scheduler_id.store(id, Ordering::Release);
    }

    /// Returns the current process group leader
    ///
    /// This function must not be called from outside the process itself
    #[inline]
    pub fn group_leader(&self) -> Option<&Pid> {
        self.guard.group_leader.as_ref()
    }

    /// Sets the current process group leader
    ///
    /// This function must not be called from outside the process itself
    pub fn set_group_leader(&mut self, gl: Pid) {
        self.guard.group_leader = Some(gl);
    }

    /// Sets the given process flags
    #[inline]
    pub fn set_flags(&mut self, flags: ProcessFlags) {
        self.guard.flags |= flags;
    }

    /// Removes the given process flags
    #[inline]
    pub fn remove_flags(&mut self, flags: ProcessFlags) {
        self.guard.flags.remove(flags);
    }

    /// Sets the process timer to `timer`, if no timer is currently set
    ///
    /// The given timer must be `Active`, use `cancel_timer` or `set_timeout` for other timer
    /// values.
    ///
    /// This has the effect of also setting the `IN_TIMER_QUEUE` process flag.
    #[inline]
    pub fn set_timer(
        &mut self,
        timer: ProcessTimer,
        timer_ref: ReferenceId,
    ) -> Result<(), ProcessTimer> {
        assert_matches!(timer, ProcessTimer::Active(_));
        self.as_ref().timer.compare_exchange(
            ProcessTimer::None,
            timer,
            Ordering::Release,
            Ordering::SeqCst,
        )?;
        self.guard.timer_ref = timer_ref;
        self.guard.flags |= ProcessFlags::IN_TIMER_QUEUE;
        Ok(())
    }

    /// Sets the process timer to `None`, returning the previous timer value
    ///
    /// This has the effect of removing the `IN_TIMER_QUEUE` and `TIMEOUT` process flags.
    #[inline]
    pub fn cancel_timer(&mut self) -> (ProcessTimer, Option<ReferenceId>) {
        let prev = self
            .as_ref()
            .timer
            .swap(ProcessTimer::None, Ordering::Release);
        self.guard
            .flags
            .remove(ProcessFlags::IN_TIMER_QUEUE | ProcessFlags::TIMEOUT);
        let timer_ref = mem::replace(&mut self.guard.timer_ref, ReferenceId::zero());
        let timer_ref = if timer_ref.is_zero() {
            None
        } else {
            Some(timer_ref)
        };
        (prev, timer_ref)
    }

    /// If the current process timer state is `Timeout`, then clear the state to `None` and return
    /// `true`
    ///
    /// Otherwise returns `false`
    pub fn clear_timer_on_timeout(&self) -> bool {
        match self.as_ref().timer.compare_exchange(
            ProcessTimer::TimedOut,
            ProcessTimer::None,
            Ordering::Release,
            Ordering::SeqCst,
        ) {
            Ok(_) => {
                assert!(self.guard.flags.contains(ProcessFlags::TIMEOUT));
                true
            }
            _ => false,
        }
    }

    /// Return true if a garbage collection is beneficial at this time
    #[inline]
    pub fn is_gc_desired(&self) -> bool {
        if self.guard.gc_needed > 0 {
            true
        } else if self.guard.flags.contains(ProcessFlags::FORCE_GC) {
            true
        } else {
            self.guard.heap.should_collect(self.guard.gc_threshold)
        }
    }

    /// Performs a garbage collection on this process
    ///
    /// In order to initiate the collection, the caller must prove that they hold
    /// the main process lock, by passing it in. This is also used to access volatile
    /// elements of the process execution state, such as its call stack and heap safely.
    ///
    /// If collection completes successfully, this function returns `Ok` with the number
    /// of reductions that the collection approximately cost. Otherwise, `Err` is returned
    /// with the cause.
    #[inline(never)]
    pub fn garbage_collect(&mut self, mut roots: RootSet) -> Result<usize, GcError> {
        let needed = self.guard.gc_needed;
        log::trace!(target: "process", "starting garbage collection ({} bytes needed)", needed);

        for root in self.guard.stack.stack.iter().take(self.guard.stack.sp) {
            roots += (root as *const OpaqueTerm).cast_mut();
        }

        if let Some(ref tuple) = unsafe { &*self.process.initial_arguments.get() } {
            roots += (tuple as *const Term).cast_mut();
        }

        let gc_count = self.guard.gc_count;
        let fullsweep_after = self.as_ref().fullsweep_after.load(Ordering::Relaxed);
        if gc_count >= fullsweep_after {
            self.guard.flags |= ProcessFlags::NEED_FULLSWEEP;
        }

        if self.guard.flags.contains(ProcessFlags::NEED_FULLSWEEP) {
            self.gc_full(needed, roots)
        } else {
            self.gc_minor(needed, roots)
        }
    }

    fn gc_full(&mut self, needed: usize, roots: RootSet) -> Result<usize, GcError> {
        use crate::gc::*;
        use firefly_alloc::heap::GenerationalHeap;

        log::trace!(target: "gc", "performing major garbage collection");

        // Determine the estimated size for the new heap
        let min_heap_size = self.as_ref().min_heap_size.map(|sz| sz.get()).unwrap_or(0);
        let mature_heap_size = self.heap.mature().heap_used();
        let size_before = self.heap.immature().heap_used() + mature_heap_size;
        log::trace!(target: "gc", "source heap size is {} bytes", size_before);
        let estimated_size = cmp::max(min_heap_size, size_before + needed);
        let baseline_size = ProcessHeap::next_size(estimated_size);
        log::trace!(target: "gc", "target baseline heap size is {} bytes", baseline_size);

        // If we already have a large enough heap, we don't need to grow it, but if the
        // HEAP_GROW flag is set, then we should do it anyway, since it will prevent us
        // from doing another full collection for awhile
        let force_grow = self.guard.flags.contains(ProcessFlags::HEAP_GROW);
        let new_heap_size = if baseline_size == self.heap.immature().heap_size() || force_grow {
            log::trace!(target: "gc", "forcing heap growth");
            ProcessHeap::next_size(baseline_size)
        } else {
            baseline_size
        };

        // Verify that our projected heap size is not going to blow the max heap size, if set
        let max_heap_size = self.max_heap_size();
        if let Some(max_size) = max_heap_size.size {
            if max_size.get() < new_heap_size {
                log::trace!(target: "gc", "max heap size of {} bytes was exceeded ({})", max_size, new_heap_size);
                return Err(GcError::MaxHeapSizeExceeded);
            }
        }

        // Unset the heap growth flag and fullswep flags
        self.guard
            .flags
            .remove(ProcessFlags::HEAP_GROW | ProcessFlags::NEED_FULLSWEEP);

        // Allocate target heap (new immature generation)
        let mut target = ProcessHeap::new(new_heap_size);

        let mut collector =
            SimpleCollector::new(FullCollection::new(&mut self.guard.heap, &mut target));
        let moved = collector.garbage_collect(roots)?;

        // Calculate reclamation for tracing
        let size_after = self.guard.heap.immature().heap_used();
        let total_size = self.guard.heap.immature().heap_size();
        let needed_after = cmp::max(min_heap_size, size_after + needed);
        if size_before >= size_after {
            log::trace!(target: "gc", "garbage collection reclaimed {} bytes", size_before - size_after);
        } else {
            log::trace!(target: "gc", "garbage collection resulted in heap growth of {} bytes", size_after - size_before);
        }

        // Check if the needed space consumes more than 75% of the new heap,
        // and if so, schedule some heap growth to try and get ahead of allocations
        // failing due to lack of space
        if total_size * 3 < needed_after * 4 {
            log::trace!(target: "gc", "little space remains on the heap after collection, requesting heap growth on next cycle");
            self.guard.flags |= ProcessFlags::HEAP_GROW;
            return Ok(estimate_cost(size_after, 0));
        }

        // Check if the needed space consumes less than 25% of the new heap,
        // and if so, shrink the new heap immediately to free the unused space
        if total_size > needed_after * 4 && ProcessHeap::DEFAULT_SIZE < total_size {
            let mut estimate = needed_after * 2;
            // If the estimated need is too low, round up to the min heap size;
            // otherwise, calculate the next heap size bucket our need falls into
            if estimate < ProcessHeap::DEFAULT_SIZE {
                estimate = ProcessHeap::DEFAULT_SIZE;
            } else {
                estimate = ProcessHeap::next_size(estimate);
            }

            // As a sanity check, only shrink the heap if the estimate is actually smaller
            if estimate < total_size {
                log::trace!(target: "gc", "the current heap is oversized, should be shrunk to {} bytes", estimate);
                // return Ok(estimate_cost(moved, size_after));
            }
        }

        self.guard.gc_count = 0;

        Ok(estimate_cost(moved, 0))
    }

    fn gc_minor(&mut self, needed: usize, roots: RootSet) -> Result<usize, GcError> {
        use crate::gc::*;
        use firefly_alloc::heap::GenerationalHeap;

        log::trace!(target: "gc", "performing minor garbage collection");

        // Determine the estimated size for the new heap
        let min_heap_size = self.as_ref().min_heap_size.map(|sz| sz.get()).unwrap_or(0);
        let size_before = self.guard.heap.immature().heap_used();
        log::trace!(target: "gc", "source heap usage is {} bytes", size_before);
        let mature_range = self.guard.heap.immature().mature_range();
        let mature_size = unsafe { mature_range.start.sub_ptr(mature_range.end) };
        log::trace!(target: "gc", "source mature heap size is {} bytes", mature_size);

        // Verify that our projected heap size does not exceed the max heap size, if set
        let max_heap_size = self.max_heap_size();
        let has_mature = !self.guard.heap.mature().is_empty();
        if let Some(max_size) = max_heap_size.size {
            let mut heap_size = size_before;
            if !has_mature && mature_size > 0 {
                heap_size += ProcessHeap::next_size(size_before);
            } else if has_mature {
                heap_size += self.guard.heap.mature().heap_used();
            }

            // Add potential new young heap size, conservatively estimating
            // the worst case scenario where we free no memory and need to reclaim
            // `needed` bytes. We grow the projected size until there is at least
            // enough memory for the current heap + `needed`
            let baseline_size = size_before + needed;
            heap_size += ProcessHeap::next_size(baseline_size);

            if heap_size > max_size.get() {
                log::trace!(target: "gc", "estimated target heap size of {} exceeds the max heap size of {}", heap_size, max_size);
                return Err(GcError::MaxHeapSizeExceeded);
            }
        }

        // Allocate an old generation if we don't have one
        if !has_mature && mature_size > 0 {
            let size = ProcessHeap::next_size(size_before);
            log::trace!(target: "gc", "allocating a fresh mature generation heap of {} bytes", size);
            let heap = ProcessHeap::new(size);
            let _ = self.guard.heap.swap_mature(heap);
        }

        // If the old heap isn't present, or isn't large enough to hold
        // the mature objects in the young generation, then a full sweep is
        // required
        let mature_available = self.guard.heap.mature().heap_available();
        if has_mature && mature_size > mature_available {
            log::trace!(target: "gc", "insufficient space on target mature heap (only {} available), full sweep required", mature_available);
            // Switch to a full collection
            return self.gc_full(needed, roots);
        }

        let prev_old_top = self.guard.heap.mature().heap_top();
        let baseline_size = cmp::max(min_heap_size, size_before + needed);
        // While we expect that we will free memory during collection,
        // we want to avoid the case where we collect and then find that
        // the new heap is too small to meet the need that triggered the
        // collection in the first place. Better to shrink it post-collection
        // than to require growing it and re-updating all the roots again
        let new_size = ProcessHeap::next_size(baseline_size);
        log::trace!(target: "gc", "new immature heap size is {} bytes", new_size);
        let target = ProcessHeap::new(new_size);

        // Swap it with the existing immature heap
        let mut source = self.guard.heap.swap_immature(target);
        let mut collector =
            SimpleCollector::new(MinorCollection::new(&mut source, &mut self.guard.heap));
        let moved = collector.garbage_collect(roots)?;

        // Calculate memory usage after collection
        let new_mature_size = unsafe { self.guard.heap.mature().heap_top().sub_ptr(prev_old_top) };
        let heap_used = self.guard.heap.immature().heap_used();
        //let size_after = new_mature_size + heap_used;
        let needed_after = cmp::max(min_heap_size, heap_used + needed);
        let heap_size = self.guard.heap.immature().heap_size();
        let mature_heap_size = self.guard.heap.mature().heap_size();
        let is_oversized = heap_size > needed_after * 4;

        log::trace!(target: "gc", "immature heap usage after gc is {} of {} bytes", heap_used, heap_size);
        log::trace!(target: "gc", "mature heap usage after gc is {} of {} bytes", new_mature_size, mature_heap_size);
        if is_oversized {
            log::trace!(target: "gc", "immature heap is oversized by at least 4x");
        }

        let should_shrink = is_oversized && (heap_size > 8000 || heap_size > mature_heap_size);
        if should_shrink {
            // We are going to shrink the heap to 3x the size of our current need,
            // at this point we already know that the heap is more than 4x our current need,
            // so this provides a reasonable baseline heap usage of 33%
            let mut estimate = needed_after * 3;
            // However, if this estimate is very small compared to the size of
            // the old generation, then we are likely going to be forced to reallocate
            // sooner rather than later, as the old generation would seem to indicate
            // that we allocate many objects.
            //
            // We determine our estimate to be too small if it is less than 10% the
            // size of the old generation. In this situation, we set our estimate to
            // be 25% of the old generation heap size
            if estimate * 9 < mature_heap_size {
                estimate = mature_heap_size / 8;
            }

            // If the new estimate is less than the min heap size, then round up;
            // otherwise, round the estimate up to the nearest heap size bucket
            if estimate < ProcessHeap::DEFAULT_SIZE {
                estimate = ProcessHeap::DEFAULT_SIZE;
            } else {
                estimate = ProcessHeap::next_size(estimate);
            }

            // As a sanity check, only shrink if our revised estimate is
            // actually smaller than the current heap size
            if estimate < heap_size {
                log::trace!(target: "gc", "should shrink immature heap to {} bytes", estimate);
                // Our final cost should account for the moved heap
                // return Ok(estimate_cost(moved, heap_used));
            }
        }

        self.guard.gc_count += 1;

        Ok(estimate_cost(moved, 0))
    }
}
unsafe impl<'a> Allocator for ProcessLock<'a> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.guard.heap.deallocate(ptr, layout)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.grow(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.grow_zeroed(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.shrink(ptr, old_layout, new_layout)
    }
}
unsafe impl<'a> Allocator for &mut ProcessLock<'a> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.guard.heap.deallocate(ptr, layout)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.grow(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.grow_zeroed(ptr, old_layout, new_layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        self.guard.heap.shrink(ptr, old_layout, new_layout)
    }
}

impl<'a> Heap for ProcessLock<'a> {
    #[inline]
    fn heap_start(&self) -> *mut u8 {
        self.guard.heap.heap_start()
    }

    #[inline]
    fn heap_top(&self) -> *mut u8 {
        self.guard.heap.heap_top()
    }

    #[inline]
    unsafe fn reset_heap_top(&self, ptr: *mut u8) {
        self.guard.heap.reset_heap_top(ptr);
    }

    #[inline]
    fn heap_end(&self) -> *mut u8 {
        self.guard.heap.heap_end()
    }

    #[inline]
    fn high_water_mark(&self) -> Option<NonNull<u8>> {
        self.guard.heap.high_water_mark()
    }

    #[inline]
    fn contains(&self, ptr: *const ()) -> bool {
        self.guard.heap.contains(ptr)
    }
}
