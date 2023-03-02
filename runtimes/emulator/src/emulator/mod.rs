mod scheduler;

use std::cell::{Cell, RefCell, UnsafeCell};
use std::ptr;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crossbeam::deque::Injector;

use firefly_alloc::fragment::HeapFragment;
use firefly_bytecode::ByteCode;
use firefly_rt::function::ModuleFunctionArity;
use firefly_rt::process::Process;
use firefly_rt::scheduler::SchedulerId;
use firefly_rt::services::{registry, timers};
use firefly_rt::term::{atom, Atom, Cons, LayoutBuilder, ReferenceId, Term};

use tokio::runtime::Handle;

use crate::queue::{LocalProcessQueue, RunQueue};

pub(crate) use self::scheduler::Action;

/// Represents a failure in the emulator during execution
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum EmulatorError {
    InvalidInit,
    SystemLimit,
    Halt(u32),
}

thread_local! {
    static CURRENT_SCHEDULER: Cell<*mut Emulator> = Cell::new(ptr::null_mut());
}

/// Returns a reference to the current scheduler on this thread
pub fn current_scheduler<'a>() -> &'a Emulator {
    unsafe { &*CURRENT_SCHEDULER.get() }
}

/// An instance of the virtual machine emulator
///
/// One of these is run for each scheduler thread on which processes are scheduled.
///
/// For other forms of tasks, a handle to the async runtime is provided which can be used
/// to spawn both async and synchronous tasks. Virtually all system tasks and I/O are performed
/// using the async runtime rather than our process schedulers.
pub struct Emulator {
    /// The current scheduler id. Set once at creation and never changes.
    id: SchedulerId,
    /// A reference to the bytecode loaded at startup. This never changes.
    code: Arc<ByteCode<Atom, atom::GlobalAtomTable>>,
    /// The scheduler run queue for processes.
    ///
    /// This queue is safe to access from multiple threads, and is designed to support
    /// work stealing to/from other schedulers. It holds a reference to the global task
    /// queue in which newly spawned processes are placed.
    runq: RunQueue<LocalProcessQueue>,
    injector: Arc<Injector<Arc<Process>>>,
    /// A handle to the async runtime
    ///
    /// This should be used for scheduling tasks which aren't backed by a process
    #[allow(unused)]
    handle: Handle,
    /// This is internal state to the scheduler, containing the value of the next unique reference
    /// id for this scheduler.
    reference_id: UnsafeCell<u64>,
    /// This is internal state to the scheduler, containing the value of the next unique integer
    /// for this scheduler.
    unique_id: UnsafeCell<u64>,
    /// This generally corresponds to the scheduler id, but is useful for checking whether an
    /// caller might be coming from the same thread or not. This value is set once when the
    /// scheduler starts and never changes after that (schedulers are not permitted to migrate
    /// threads).
    thread_id: std::thread::ThreadId,
    /// The total reduction count executed by this scheduler
    reductions: AtomicU64,
    /// This is internal state to the scheduler, providing the timer service for processes
    /// scheduled on this scheduler
    ///
    /// Processes which are suspended on a timer are not in any task queue, but in other cases
    /// where a process is linked to a timer and it gets migrated to another scheduler, the
    /// scheduler which owns the timer will relay the event to the owning scheduler. Likewise,
    /// if a scheduler takes possession of a process and a request to cancel a timer is
    /// received, it will look up the scheduler id in the timer reference and relay the
    /// cancellation to the scheduler on which the timer was registered.
    timers: RefCell<timers::PerSchedulerTimerService>,
}
unsafe impl Send for Emulator {}
unsafe impl Sync for Emulator {}
impl Emulator {
    /// Create a new [`EmulatorThread`] for the given bytecode module
    pub fn new(
        id: SchedulerId,
        code: Arc<ByteCode<Atom, atom::GlobalAtomTable>>,
        injector: Arc<Injector<Arc<Process>>>,
        handle: Handle,
    ) -> Arc<Self> {
        let runq = RunQueue::new(injector.clone());
        Arc::new(Self {
            id,
            code,
            runq,
            injector,
            handle,
            reference_id: UnsafeCell::new(ReferenceId::init()),
            unique_id: UnsafeCell::new(0),
            thread_id: std::thread::current().id(),
            reductions: AtomicU64::new(0),
            timers: RefCell::new(timers::PerSchedulerTimerService::new()),
        })
    }

    /// This function starts the scheduler loop of this emulator, returning a join handle
    /// which can be used to await the exit status of the emulator from the calling thread.
    pub fn start(self: Arc<Self>, spawn_init: bool) -> Result<(), EmulatorError> {
        let ptr = Arc::as_ptr(&self);
        assert_eq!(
            CURRENT_SCHEDULER.replace(ptr.cast_mut()),
            ptr::null_mut(),
            "cannot run two schedulers on the same thread!"
        );

        // Spawn the init process before first run
        if spawn_init {
            unsafe {
                self.spawn_init()?;
            }
        }

        // Start the scheduler control loop
        self.run()
    }

    /// # SAFETY
    ///
    /// This function must only be called once, and only on one scheduler in the system, otherwise
    /// I suspect all hell will break loose.
    unsafe fn spawn_init(&self) -> Result<(), EmulatorError> {
        use crate::queue::TaskQueue;

        // Fetch offset of `init:start/0` in the loaded bytecode
        let init = "init:boot/1".parse::<ModuleFunctionArity>().unwrap();
        let init_mfa = init.into();
        let init_offset = self
            .code
            .function_by_mfa(&init_mfa)
            .and_then(|f| f.offset())
            .ok_or(EmulatorError::InvalidInit)?;

        // Spawn with the initial arguments vector as the sole argument
        let initial_args = crate::sys::env::argv();
        let mut layout = LayoutBuilder::new();
        layout.build_list(initial_args.len());
        let fragment = HeapFragment::new(layout.finish(), None).unwrap();
        let initial_args = Cons::from_slice(initial_args, unsafe { fragment.as_ref() })
            .unwrap()
            .map(Term::Cons)
            .unwrap_or(Term::Nil);

        // Initialize fresh process state
        let mut init_p = Process::new(
            self.id,
            None,
            None,
            init,
            &[initial_args.into()],
            self.injector.clone(),
            Default::default(),
        );
        {
            let proc = Arc::get_mut(&mut init_p).unwrap();
            proc.set_instruction_pointer(init_offset);
        }

        // Register the process in the global registry
        registry::register_process(init_p.clone());

        // Schedule the process for execution immediately
        self.runq.push(init_p);

        Ok(())
    }
}
