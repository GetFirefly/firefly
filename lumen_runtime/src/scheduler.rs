use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use crate::code::Code;
#[cfg(test)]
use crate::process;
use crate::process::local::put_pid_to_process;
use crate::process::Process;
use crate::reference;
use crate::run;
use crate::run::Run;
use crate::term::Term;
use crate::timer::Hierarchy;

mod id;

#[derive(Clone, Eq, Hash, PartialEq)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ID(id::Raw);

impl ID {
    fn new(raw: id::Raw) -> ID {
        ID(raw)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Max,
}

impl Default for Priority {
    fn default() -> Priority {
        Priority::Normal
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
            SCHEDULERS
                .lock()
                .unwrap()
                .scheduler_by_id
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

    pub fn next_reference_number(&self) -> reference::local::Number {
        self.reference_count.fetch_add(1, Ordering::SeqCst)
    }

    pub fn next_reference(&self, process: &Process) -> &'static reference::local::Reference {
        self.reference(self.next_reference_number(), process)
    }

    pub fn reference(
        &self,
        number: reference::local::Number,
        process: &Process,
    ) -> &'static reference::local::Reference {
        process.local_reference(&self.id, number)
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
            self.run_once();
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
    fn run_once(&self) {
        self.hierarchy.write().unwrap().timeout(&self.id);

        loop {
            // separate from `match` below so that WriteGuard temporary is not held while process
            // runs.
            let run = self.run_queues.write().unwrap().dequeue();

            match run {
                Run::Now(arc_process) => {
                    Process::run(&arc_process);

                    match self.run_queues.write().unwrap().requeue(arc_process) {
                        // TODO exit logging
                        Some(_exiting_arc_process) => {}
                        None => (),
                    };

                    break;
                }
                Run::Delayed => continue,
                // TODO steal processes or sleep if nothing to steal
                Run::None => break,
            }
        }
    }

    pub fn run_queues_len(&self) -> usize {
        self.run_queues.read().unwrap().len()
    }

    #[cfg(test)]
    pub fn run_queue_len(&self, priority: Priority) -> usize {
        self.run_queues.read().unwrap().run_queue_len(priority)
    }

    pub fn run_through(&self, arc_process: &Arc<Process>) {
        let ordering = Ordering::SeqCst;
        let reductions_before = arc_process.total_reductions.load(ordering);

        // The same as `run`, but stops when the process is run once
        while arc_process.total_reductions.load(Ordering::SeqCst) <= reductions_before {
            self.run_once();
        }
    }

    /// `arguments` are put in reverse order on the stack where `code` can use them
    pub fn spawn(
        parent_process: &Process,
        module: Term,
        function: Term,
        arguments: Term,
        code: Code,
    ) -> Arc<Process> {
        let process = Process::spawn(parent_process, module, function, arguments, code);
        let parent_locked_option_weak_scheduler = parent_process.scheduler.lock().unwrap();

        let arc_scheduler = match *parent_locked_option_weak_scheduler {
            Some(ref weak_scheduler) => match weak_scheduler.upgrade() {
                Some(arc_scheduler) => arc_scheduler,
                None => {
                    #[cfg(debug_assertions)]
                    panic!(
                        "Parent process ({:?}) Scheduler has been Dropped",
                        parent_process
                    );
                    #[cfg(not(debug_assertions))]
                    panic!("Parent process Scheduler has been Dropped");
                }
            },
            None => {
                #[cfg(debug_assertions)]
                panic!(
                    "Parent process ({:?}) is not assigned to a Scheduler",
                    parent_process
                );
                #[cfg(not(debug_assertions))]
                panic!("Parent process is not assigned to a Scheduler");
            }
        };

        let mut writable_run_queues = arc_scheduler.run_queues.write().unwrap();

        {
            let mut locked_option_weak_scheduler = process.scheduler.lock().unwrap();
            *locked_option_weak_scheduler = Some(Arc::downgrade(&arc_scheduler));
        }

        let arc_process = Arc::new(process);

        writable_run_queues.enqueue(Arc::clone(&arc_process));

        put_pid_to_process(&arc_process);

        arc_process
    }

    pub fn spawn_init(self: Arc<Scheduler>) -> Arc<Process> {
        let arc_process = Arc::new(Process::init());
        let scheduler_arc_process = Arc::clone(&arc_process);

        // `parent_process.scheduler.lock` has to be taken first to even get the run queue in
        // `spawn`, so copy that lock order here.
        let mut locked_option_weak_scheduler = scheduler_arc_process.scheduler.lock().unwrap();
        let mut writable_run_queues = self.run_queues.write().unwrap();

        *locked_option_weak_scheduler = Some(Arc::downgrade(&self));
        writable_run_queues.enqueue(Arc::clone(&arc_process));

        put_pid_to_process(&arc_process);

        arc_process
    }

    pub fn stop_waiting(&self, process: &Process) {
        self.run_queues.write().unwrap().stop_waiting(process);
    }

    // Private

    fn new(raw_id: id::Raw) -> Scheduler {
        Scheduler {
            id: ID::new(raw_id),
            hierarchy: Default::default(),
            reference_count: AtomicU64::new(0),
            run_queues: Default::default(),
        }
    }

    fn registered() -> Arc<Scheduler> {
        let mut locked_schedulers = SCHEDULERS.lock().unwrap();
        let raw_id = locked_schedulers.id_manager.alloc();
        let arc_scheduler = Arc::new(Scheduler::new(raw_id));

        if let Some(_) = locked_schedulers
            .scheduler_by_id
            .insert(arc_scheduler.id.clone(), Arc::downgrade(&arc_scheduler))
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

impl Drop for Scheduler {
    fn drop(&mut self) {
        let mut locked_schedulers = SCHEDULERS.lock().unwrap();

        locked_schedulers
            .scheduler_by_id
            .remove(&self.id)
            .expect("Scheduler not registered");
        // Free the ID only after it is not registered to prevent manager re-allocating the ID to a
        // Scheduler for a new thread.
        locked_schedulers.id_manager.free(self.id.0)
    }
}

impl PartialEq for Scheduler {
    fn eq(&self, other: &Scheduler) -> bool {
        self.id == other.id
    }
}

thread_local! {
  static SCHEDULER: Arc<Scheduler> = Scheduler::registered();
}

// A single struct, so that one `Mutex` can protect both the `id_manager` and `scheduler_by_id`, so
// that schedulers are deregistered from `scheduler_by_id` before giving the `ID` back to the
// `manager`, which is the opposite order from creation.  The opposite order means it doesn't work
// to implement separate `Mutex`es with Drop for both `ID` and `Scheduler`.
struct Schedulers {
    id_manager: id::Manager,
    // Schedulers are `Weak` so that when the spawning thread ends, `SCHEDULER` can be dropped,
    // which will remove the entry here.
    scheduler_by_id: HashMap<ID, Weak<Scheduler>>,
}

impl Schedulers {
    fn new() -> Schedulers {
        Schedulers {
            id_manager: id::Manager::new(),
            scheduler_by_id: Default::default(),
        }
    }
}

lazy_static! {
    static ref SCHEDULERS: Mutex<Schedulers> = Mutex::new(Schedulers::new());
}

#[cfg(test)]
pub fn with_process<F>(f: F)
where
    F: FnOnce(&Process) -> (),
{
    f(&process::local::test(&process::local::test_init()))
}

#[cfg(test)]
pub fn with_process_arc<F>(f: F)
where
    F: FnOnce(Arc<Process>) -> (),
{
    f(process::local::test(&process::local::test_init()))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod scheduler {
        use super::*;

        mod new_process {
            use super::*;

            use crate::atom::Existence::DoNotCare;
            use crate::code;
            use crate::otp::erlang;

            #[test]
            fn different_processes_have_different_pids() {
                let erlang = Term::str_to_atom("erlang", DoNotCare).unwrap();
                let exit = Term::str_to_atom("exit", DoNotCare).unwrap();
                let normal = Term::str_to_atom("normal", DoNotCare).unwrap();
                let parent_arc_process = process::local::test_init();

                let first_process_arguments = Term::slice_to_list(&[normal], &parent_arc_process);
                let first_process = Scheduler::spawn(
                    &parent_arc_process,
                    erlang,
                    exit,
                    first_process_arguments,
                    code::apply_fn(),
                );

                let second_process_arguments = Term::slice_to_list(&[normal], &parent_arc_process);
                let second_process = Scheduler::spawn(
                    &first_process,
                    erlang,
                    exit,
                    second_process_arguments,
                    code::apply_fn(),
                );

                assert_ne!(
                    erlang::self_0(&first_process),
                    erlang::self_0(&second_process)
                );
            }
        }
    }
}
