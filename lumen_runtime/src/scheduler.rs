#[cfg(test)]
pub mod test;

use core::fmt::{self, Debug};
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::sync::{Arc, Weak};

use hashbrown::HashMap;

use liblumen_core::locks::{Mutex, RwLock};

use liblumen_alloc::erts::exception::system::{Alloc, Exception};
use liblumen_alloc::erts::process::code::Code;
#[cfg(test)]
use liblumen_alloc::erts::process::Priority;
use liblumen_alloc::erts::process::{Process, Status};
pub use liblumen_alloc::erts::scheduler::{id, ID};
use liblumen_alloc::erts::term::{reference, Atom, Reference, Term};

use crate::process;
use crate::process::spawn;
use crate::process::spawn::options::{Connection, Options};
use crate::registry::put_pid_to_process;
use crate::run::{self, Run};
use crate::timer::Hierarchy;

pub trait Scheduled {
    fn scheduler(&self) -> Option<Arc<Scheduler>>;
}

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

pub struct Scheduler {
    pub id: ID,
    pub hierarchy: RwLock<Hierarchy>,
    // References are always 64-bits even on 32-bit platforms
    reference_count: AtomicU64,
    run_queues: RwLock<run::queues::Queues>,
}

impl Scheduler {
    pub fn current() -> Arc<Scheduler> {
        SCHEDULER.with(|thread_local_scheduler| thread_local_scheduler.clone())
    }

    pub fn from_id(id: &ID) -> Option<Arc<Scheduler>> {
        Self::current_from_id(id).or_else(|| {
            SCHEDULER_BY_ID
                .lock()
                .get(id)
                .and_then(|arc_scheduler| arc_scheduler.upgrade())
        })
    }

    fn current_from_id(id: &ID) -> Option<Arc<Scheduler>> {
        SCHEDULER.with(|thread_local_scheduler| {
            if &thread_local_scheduler.id == id {
                Some(thread_local_scheduler.clone())
            } else {
                None
            }
        })
    }

    pub fn next_reference_number(&self) -> reference::Number {
        self.reference_count.fetch_add(1, Ordering::SeqCst)
    }

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
    pub fn run_once(&self) -> bool {
        self.hierarchy.write().timeout();

        loop {
            // separate from `match` below so that WriteGuard temporary is not held while process
            // runs.
            let run = self.run_queues.write().dequeue();

            match run {
                Run::Now(arc_process) => {
                    // Don't allow exiting processes to run again.
                    //
                    // Without this check, a process.exit() from outside the process during WAITING
                    // will return to the Frame that called `process.wait()`
                    if !arc_process.is_exiting() {
                        match Process::run(&arc_process) {
                            Ok(()) => (),
                            Err(exception) => match exception {
                                Exception::Alloc(_inner) => {
                                    match arc_process.garbage_collect(0, &mut []) {
                                        Ok(_freed) => (),
                                        Err(gc_err) => panic!("Gc error: {:?}", gc_err),
                                    }
                                }
                            },
                        }
                    } else {
                        arc_process.reduce()
                    }

                    match self.run_queues.write().requeue(arc_process) {
                        Some(exiting_arc_process) => match *exiting_arc_process.status.read() {
                            Status::Exiting(ref exception) => {
                                process::log_exit(&exiting_arc_process, exception);
                                process::propagate_exit(&exiting_arc_process, exception);
                            }
                            _ => unreachable!(),
                        },
                        None => (),
                    };

                    break true;
                }
                Run::Delayed => continue,
                // TODO steal processes or sleep if nothing to steal
                Run::None => break false,
            }
        }
    }

    pub fn run_queues_len(&self) -> usize {
        self.run_queues.read().len()
    }

    #[cfg(test)]
    pub fn run_queue_len(&self, priority: Priority) -> usize {
        self.run_queues.read().run_queue_len(priority)
    }

    #[cfg(test)]
    pub fn is_run_queued(&self, value: &Arc<Process>) -> bool {
        self.run_queues.read().contains(value)
    }

    /// Returns `true` if `arc_process` was run; otherwise, `false`.
    #[must_use]
    pub fn run_through(&self, arc_process: &Arc<Process>) -> bool {
        let ordering = Ordering::SeqCst;
        let reductions_before = arc_process.total_reductions.load(ordering);

        // The same as `run`, but stops when the process is run once
        loop {
            if self.run_once() {
                if reductions_before < arc_process.total_reductions.load(Ordering::SeqCst) {
                    break true;
                } else {
                    continue;
                }
            } else {
                break false;
            }
        }
    }

    pub fn schedule(self: Arc<Scheduler>, process: Process) -> Arc<Process> {
        let mut writable_run_queues = self.run_queues.write();

        process.schedule_with(self.id);

        let arc_process = Arc::new(process);

        writable_run_queues.enqueue(Arc::clone(&arc_process));

        arc_process
    }

    /// Spawns a process with arguments for `apply(module, function, arguments)` on its stack.
    ///
    /// This allows the `apply/3` code to be changed with `apply_3::set_code(code)` to handle new
    /// MFA unique to a given application.
    pub fn spawn_apply_3(
        parent_process: &Process,
        options: Options,
        module: Atom,
        function: Atom,
        arguments: Term,
    ) -> Result<Spawned, Alloc> {
        let spawn::Spawned {
            process,
            connection,
        } = process::spawn::apply_3(parent_process, options, module, function, arguments)?;
        let arc_scheduler = parent_process.scheduler().unwrap();
        let arc_process = arc_scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        Ok(Spawned {
            arc_process,
            connection,
        })
    }

    /// Spawns a process with `arguments` on its stack and `code` run with those arguments instead
    /// of passing through `apply/3`.
    pub fn spawn_code(
        parent_process: &Process,
        options: Options,
        module: Atom,
        function: Atom,
        arguments: &[Term],
        code: Code,
    ) -> Result<Spawned, Alloc> {
        let spawn::Spawned {
            process,
            connection,
        } = process::spawn::code(
            Some(parent_process),
            options,
            module,
            function,
            arguments,
            code,
        )?;
        let arc_scheduler = parent_process.scheduler().unwrap();
        let arc_process = arc_scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        Ok(Spawned {
            arc_process,
            connection,
        })
    }

    pub fn spawn_init(
        self: Arc<Scheduler>,
        minimum_heap_size: usize,
    ) -> Result<Arc<Process>, Alloc> {
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

    pub fn stop_waiting(&self, process: &Process) {
        self.run_queues.write().stop_waiting(process);
    }

    // Private

    fn new() -> Scheduler {
        Scheduler {
            id: id::next(),
            hierarchy: Default::default(),
            reference_count: AtomicU64::new(0),
            run_queues: Default::default(),
        }
    }

    fn registered() -> Arc<Scheduler> {
        let mut locked_scheduler_by_id = SCHEDULER_BY_ID.lock();
        let arc_scheduler = Arc::new(Scheduler::new());

        if let Some(_) =
            locked_scheduler_by_id.insert(arc_scheduler.id.clone(), Arc::downgrade(&arc_scheduler))
        {
            #[cfg(debug_assertions)]
            panic!(
                "Scheduler already registered with ID ({:?}",
                arc_scheduler.id
            );
            #[cfg(not(debug_assertions))]
            panic!("Scheduler already registered");
        }

        arc_scheduler
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
        let mut locked_scheduler_by_id = SCHEDULER_BY_ID.lock();

        locked_scheduler_by_id
            .remove(&self.id)
            .expect("Scheduler not registered");
    }
}

impl PartialEq for Scheduler {
    fn eq(&self, other: &Scheduler) -> bool {
        self.id == other.id
    }
}

pub struct Spawned {
    pub arc_process: Arc<Process>,
    #[must_use]
    pub connection: Connection,
}

impl Spawned {
    pub fn to_term(&self, process: &Process) -> Result<Term, Alloc> {
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
  static SCHEDULER: Arc<Scheduler> = Scheduler::registered();
}

lazy_static! {
    static ref SCHEDULER_BY_ID: Mutex<HashMap<ID, Weak<Scheduler>>> =
        Mutex::new(Default::default());
}

#[cfg(test)]
pub fn with_process<F>(f: F)
where
    F: FnOnce(&Process) -> (),
{
    f(&process::test(&process::test_init()))
}

#[cfg(test)]
pub fn with_process_arc<F>(f: F)
where
    F: FnOnce(Arc<Process>) -> (),
{
    f(process::test(&process::test_init()))
}
