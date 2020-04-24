#[cfg(test)]
pub mod test;

use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::SystemException;
use liblumen_alloc::erts::process::{Priority, Process, Status};
pub use liblumen_alloc::erts::scheduler::{id, ID};
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::process::{log_exit, propagate_exit, CURRENT_PROCESS};
use lumen_rt_core::registry::put_pid_to_process;
pub use lumen_rt_core::scheduler::{
    current, from_id, run_through, Scheduled, SchedulerDependentAlloc, Spawned,
};
use lumen_rt_core::scheduler::{
    run_queue, set_unregistered, unregister, Run, Scheduler as SchedulerTrait,
};
use lumen_rt_core::timer::Hierarchy;

use crate::process;
use liblumen_alloc::Ran;

fn unregistered() -> Arc<dyn lumen_rt_core::scheduler::Scheduler> {
    Arc::new(Scheduler {
        id: id::next(),
        hierarchy: Default::default(),
        reference_count: AtomicU64::new(0),
        run_queues: Default::default(),
        unique_integer: AtomicU64::new(0),
    })
}

pub fn set_unregistered_once() {
    use std::sync::Once;
    static SET_UNREGISTERED: Once = Once::new();
    SET_UNREGISTERED.call_once(|| set_unregistered(Box::new(unregistered)))
}

pub struct Scheduler {
    pub id: ID,
    pub hierarchy: RwLock<Hierarchy>,
    // References are always 64-bits even on 32-bit platforms
    reference_count: AtomicU64,
    run_queues: RwLock<run_queue::Queues>,
    // Non-monotonic unique integers are scoped to the scheduler ID and then use this per-scheduler
    // `u64`.
    unique_integer: AtomicU64,
}

impl Scheduler {
    /// > 1. Update reduction counters
    /// > 2. Check timers
    /// > 3. If needed check balance
    /// > 4. If needed migrated processes and ports
    /// > 5. Do auxiliary scheduler work
    /// > 6. If needed check I/O and update time
    /// > 7. While needed pick a port task to execute
    /// > 8. Pick a process to execute
    /// > -- [The Scheduler Loop](https://blog.stenmans.org/theBeamBook/#_the_scheduler_loop)
    pub fn run(&self) {
        loop {
            // TODO sleep or steal if nothing run
            let _ = self.run_once();
        }
    }

    pub fn is_run_queued(&self, value: &Arc<Process>) -> bool {
        self.run_queues.read().contains(value)
    }
}

impl Debug for Scheduler {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Scheduler")
            .field("id", &self.id)
            // The hiearchy slots take a lot of space, so don't print them by default
            .field("reference_count", &self.reference_count)
            .field("run_queues", &self.run_queues)
            .finish()
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        unregister(&self.id);
    }
}

impl PartialEq for Scheduler {
    fn eq(&self, other: &Scheduler) -> bool {
        self.id == other.id
    }
}

impl SchedulerTrait for Scheduler {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ID {
        self.id
    }

    fn hierarchy(&self) -> &RwLock<Hierarchy> {
        &self.hierarchy
    }

    fn next_reference_number(&self) -> ReferenceNumber {
        self.reference_count.fetch_add(1, Ordering::SeqCst)
    }

    fn next_unique_integer(&self) -> u64 {
        self.unique_integer.fetch_add(1, Ordering::SeqCst)
    }

    fn run_once(&self) -> bool {
        self.hierarchy.write().timeout();

        loop {
            // separate from `match` below so that WriteGuard temporary is not held while process
            // runs.
            let run = self.run_queues.write().dequeue();

            match run {
                Run::Now(arc_process) => {
                    CURRENT_PROCESS
                        .with(|current_process| current_process.replace(Some(arc_process.clone())));

                    // Don't allow exiting processes to run again.
                    //
                    // Without this check, a process.exit() from outside the process during WAITING
                    // will return to the Frame that called `process.wait()`
                    if !arc_process.is_exiting() {
                        match arc_process.run() {
                            Ran::Waiting | Ran::Reduced | Ran::RuntimeException => (),
                            Ran::SystemException => match &*arc_process.status.read() {
                                Status::SystemException(system_exception) => match system_exception
                                {
                                    SystemException::Alloc(_) => {
                                        match arc_process.garbage_collect(0, &mut []) {
                                            Ok(_freed) => (),
                                            Err(gc_err) => panic!(
                                                "fatal garbage collection error: {:?}",
                                                gc_err
                                            ),
                                        }
                                    }
                                    err => panic!("system error: {}", err),
                                },
                                _ => unreachable!(),
                            },
                        }
                    } else {
                        arc_process.reduce()
                    }

                    match self.run_queues.write().requeue(arc_process) {
                        Some(exiting_arc_process) => match *exiting_arc_process.status.read() {
                            Status::RuntimeException(ref exception) => {
                                log_exit(&exiting_arc_process, exception);
                                propagate_exit(&exiting_arc_process, exception);
                            }
                            _ => unreachable!(),
                        },
                        None => (),
                    };

                    CURRENT_PROCESS.with(|current_process| current_process.replace(None));

                    break true;
                }
                Run::Delayed => continue,
                // TODO steal processes or sleep if nothing to steal
                Run::None => break false,
            }
        }
    }

    fn run_queue_len(&self, priority: Priority) -> usize {
        self.run_queues.read().run_queue_len(priority)
    }

    fn run_queues_len(&self) -> usize {
        self.run_queues.read().len()
    }

    fn schedule(&self, process: Process) -> Arc<Process> {
        assert_eq!(*process.status.read(), Status::Runnable);
        let mut writable_run_queues = self.run_queues.write();

        process.schedule_with(self.id);

        let arc_process = Arc::new(process);

        writable_run_queues.enqueue(Arc::clone(&arc_process));

        arc_process
    }

    // TODO: Request application master termination for controlled shutdown
    // This request will always come from the thread which spawned the application
    // master, i.e. the "main" scheduler thread
    //
    // Returns `Ok(())` if shutdown was successful, `Err(anyhow::Error)` if something
    // went wrong during shutdown, and it was not able to complete normally
    fn shutdown(&self) -> anyhow::Result<()> {
        // For now just Ok(()), but this needs to be addressed when proper
        // system startup/shutdown is in place
        Ok(())
    }

    fn spawn_init(&self, minimum_heap_size: usize) -> Result<Arc<Process>, SystemException> {
        let process = process::init(minimum_heap_size)?;
        let arc_process = Arc::new(process);
        let scheduler_arc_process = Arc::clone(&arc_process);

        // `parent_process.scheduler.lock` has to be taken first to even get the run queue in
        // `spawn`, so copy that lock order here.
        scheduler_arc_process.schedule_with(self.id);
        let mut writable_run_queues = self.run_queues.write();

        writable_run_queues.enqueue(Arc::clone(&arc_process));

        put_pid_to_process(&arc_process);

        Ok(arc_process)
    }

    fn stop_waiting(&self, process: &Process) {
        self.run_queues.write().stop_waiting(process);
    }
}
