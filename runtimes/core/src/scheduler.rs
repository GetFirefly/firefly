pub mod run_queue;

use std::any::Any;
use std::fmt::Debug;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

use hashbrown::HashMap;

use lazy_static::lazy_static;

use liblumen_core::locks::{Mutex, RwLock};

use liblumen_alloc::erts::exception::{self, AllocResult, SystemException};
use liblumen_alloc::erts::process::Process;
pub use liblumen_alloc::erts::scheduler::id::ID;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Priority;

use crate::process::spawn::options::Connection;
use crate::timer::Hierarchy;

pub fn current() -> Arc<dyn Scheduler> {
    SCHEDULER.with(|thread_local_scheduler| thread_local_scheduler.clone())
}

fn current_from_id(id: &ID) -> Option<Arc<dyn Scheduler>> {
    SCHEDULER.with(|thread_local_scheduler| {
        if &thread_local_scheduler.id() == id {
            Some(thread_local_scheduler.clone())
        } else {
            None
        }
    })
}

pub fn from_id(id: &ID) -> Option<Arc<dyn Scheduler>> {
    current_from_id(id).or_else(|| {
        SCHEDULER_BY_ID
            .lock()
            .get(id)
            .and_then(|arc_scheduler| arc_scheduler.upgrade())
    })
}

pub fn set_unregistered(unregistered: Box<dyn Fn() -> Arc<dyn Scheduler> + 'static + Sync + Send>) {
    *RW_LOCK_OPTION_UNREGISTERED.write() = Some(unregistered);
}

fn unregistered() -> Arc<dyn Scheduler> {
    match &*RW_LOCK_OPTION_UNREGISTERED.read() {
        Some(unregistered) => unregistered(),
        None => {
            panic!("scheduler::set_unregistered not called before calling scheduler::unregistered")
        }
    }
}

fn registered() -> Arc<dyn Scheduler> {
    let mut locked_scheduler_by_id = SCHEDULER_BY_ID.lock();
    let arc_scheduler = unregistered();

    if let Some(_) =
        locked_scheduler_by_id.insert(arc_scheduler.id().clone(), Arc::downgrade(&arc_scheduler))
    {
        #[cfg(debug_assertions)]
        panic!(
            "Scheduler already registered with ID ({:?}",
            arc_scheduler.id()
        );
        #[cfg(not(debug_assertions))]
        panic!("Scheduler already registered");
    }

    arc_scheduler
}

pub fn unregister(id: &ID) {
    let mut locked_scheduler_by_id = SCHEDULER_BY_ID.lock();

    locked_scheduler_by_id
        .remove(id)
        .expect("Scheduler not registered");
}

/// Returns `true` if `arc_process` was run; otherwise, `false`.
#[must_use]
pub fn run_through(process: &Process) -> bool {
    assert!(
        !process.is_exiting(),
        "Process ({}) is exiting ({:?}) and so can't be run through",
        process,
        process.status.read()
    );

    let scheduler = process.scheduler().unwrap();
    let ordering = Ordering::SeqCst;
    let reductions_before = process.total_reductions.load(ordering);

    // The same as `run`, but stops when the process is run once
    loop {
        if scheduler.run_once() {
            if reductions_before < process.total_reductions.load(Ordering::SeqCst) {
                break true;
            } else {
                continue;
            }
        } else {
            break false;
        }
    }
}

/// What to run
pub enum Run {
    /// Run the process now
    Now(Arc<Process>),
    /// There was a process in the queue, but it needs to be delayed because it is `Priority::Low`
    /// and hadn't been delayed enough yet.  Ask the `RunQueue` again for another process.
    /// -- https://github.com/erlang/otp/blob/fe2b1323a3866ed0a9712e9d12e1f8f84793ec47/erts/emulator/beam/erl_process.c#L9601-L9606
    Delayed,
    /// There are no processes in the run queue, do other work
    None,
}

pub trait Scheduled {
    fn scheduler(&self) -> Option<Arc<dyn Scheduler>>;
}

impl Scheduled for Process {
    fn scheduler(&self) -> Option<Arc<dyn Scheduler>> {
        self.scheduler_id()
            .and_then(|scheduler_id| from_id(&scheduler_id))
    }
}

impl Scheduled for Reference {
    fn scheduler(&self) -> Option<Arc<dyn Scheduler>> {
        from_id(&self.scheduler_id())
    }
}

pub trait Scheduler: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ID;
    fn hierarchy(&self) -> &RwLock<Hierarchy>;
    fn next_reference_number(&self) -> ReferenceNumber;

    /// Gets the next available unique integer
    fn next_unique_integer(&self) -> u64;

    /// > 1. Update reduction counters
    /// > 2. Check timers
    /// > 3. If needed check balance
    /// > 4. If needed migrated processes and ports
    /// > 5. Do auxiliary scheduler work
    /// > 6. If needed check I/O and update time
    /// > 7. While needed pick a port task to execute
    /// > 8. Pick a process to execute
    /// > -- [The Scheduler Loop](https://blog.stenmans.org/theBeamBook/#_the_scheduler_loop)
    ///
    /// Returns `true` if a process was run.  Returns `false` if no process could be run and the
    /// scheduler should sleep or work steal.
    #[must_use]
    fn run_once(&self) -> bool;
    fn run_queue_len(&self, priority: Priority) -> usize;
    /// Returns the length of the current scheduler's run queue
    fn run_queues_len(&self) -> usize;
    /// Schedules the given process for execution
    fn schedule(&self, process: Process) -> Arc<Process>;
    /// Spawns the init process, should be called immediately after
    /// (primary) scheduler creation.
    fn spawn_init(&self, minimum_heap_size: usize) -> Result<Arc<Process>, SystemException>;
    fn shutdown(&self) -> anyhow::Result<()>;
    fn stop_waiting(&self, process: &Process);
}

pub trait SchedulerDependentAlloc {
    fn next_reference(&self) -> AllocResult<Term>;
}

impl SchedulerDependentAlloc for Process {
    fn next_reference(&self) -> AllocResult<Term> {
        let scheduler_id = self.scheduler_id().unwrap();
        let arc_scheduler = from_id(&scheduler_id).unwrap();
        let number = arc_scheduler.next_reference_number();

        self.reference_from_scheduler(scheduler_id, number)
    }
}

pub struct Spawned {
    pub arc_process: Arc<Process>,
    #[must_use]
    pub connection: Connection,
}

impl Spawned {
    pub fn to_term(&self, process: &Process) -> exception::Result<Term> {
        let pid_term = self.arc_process.pid_term();

        match self.connection.monitor_reference {
            Some(monitor_reference) => process
                .tuple_from_slice(&[pid_term, monitor_reference])
                .map_err(|alloc| alloc.into()),
            None => Ok(pid_term),
        }
    }
}

thread_local! {
  static SCHEDULER: Arc<dyn Scheduler> = registered();
}

lazy_static! {
    static ref RW_LOCK_OPTION_UNREGISTERED: RwLock<Option<Box<dyn Fn() -> Arc<dyn Scheduler> + 'static + Sync + Send>>> =
        RwLock::new(None);
    static ref SCHEDULER_BY_ID: Mutex<HashMap<ID, Weak<dyn Scheduler>>> =
        Mutex::new(Default::default());
}
