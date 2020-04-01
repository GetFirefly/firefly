pub mod run_queue;

use std::sync::Arc;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::process::Process;
pub use liblumen_alloc::erts::scheduler::id::ID;
use liblumen_alloc::erts::term::prelude::ReferenceNumber;

use crate::timer::Hierarchy;

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

pub trait Scheduler {
    fn id(&self) -> ID;

    /// Gets the current thread's timer wheel
    fn hierarchy(&self) -> &RwLock<Hierarchy>;

    /// Gets the next available reference number
    fn next_reference_number(&self) -> ReferenceNumber;
}

pub trait Scheduled {
    type Scheduler;

    fn scheduler(&self) -> Option<Arc<Self::Scheduler>>;
}

/*
impl Scheduled for Process {
    fn scheduler(&self) -> Option<Arc<Scheduler>> {
        self.scheduler_id()
            .and_then(|scheduler_id| Scheduler::from_id(&scheduler_id))
    }
}

impl Scheduled for Reference {
    fn scheduler(&self) -> Option<Arc<Scheduler>> {
        Scheduler::from_id(&self.scheduler_id())
    }
}
*/
