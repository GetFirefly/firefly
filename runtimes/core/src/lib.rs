pub mod context;
pub mod proplist;
pub mod registry;
pub mod time;
pub mod timer;

use std::sync::Arc;

use liblumen_alloc::erts::scheduler::id;
use liblumen_alloc::erts::term::prelude::ReferenceNumber;
use liblumen_core::locks::RwLock;

use self::timer::Hierarchy;

pub trait Scheduler {
    fn id(&self) -> id::ID;

    /// Gets the current thread's scheduler
    fn current() -> Arc<Self>;

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
