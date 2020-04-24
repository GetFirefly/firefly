pub mod options;

use std::sync::Arc;

use liblumen_alloc::erts::process::Process;

use crate::scheduler::{self, Scheduled, Scheduler};

pub use self::options::{Connection, Options};
use crate::registry::put_pid_to_process;

pub struct Spawned {
    pub process: Process,
    #[must_use]
    pub connection: Connection,
}

impl Spawned {
    pub fn schedule_with_parent(self, parent: &Process) -> scheduler::Spawned {
        self.schedule_with_scheduler(parent.scheduler().unwrap())
    }

    pub fn schedule_with_scheduler(self, scheduler: Arc<dyn Scheduler>) -> scheduler::Spawned {
        let Self {
            process,
            connection,
        } = self;
        let arc_process = scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        scheduler::Spawned {
            arc_process,
            connection,
        }
    }
}
