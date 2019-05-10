use std::collections::vec_deque::VecDeque;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

#[cfg(test)]
use crate::process;
use crate::process::local::put_pid_to_process;
use crate::process::Process;
#[cfg(test)]
use crate::process::Status;
use crate::reference;
use crate::term::Term;
use crate::timer::Hierarchy;

mod id;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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

#[derive(Default)]
pub struct RunQueue(VecDeque<Arc<Process>>);

impl RunQueue {
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn push_back(&mut self, process: Arc<Process>) {
        self.0.push_back(process);
    }

    #[cfg(test)]
    fn run_through(&mut self, process: &Arc<Process>) {
        let pid = process.pid;

        while let Some(front_arc_process) = self.0.pop_front() {
            Process::run(&front_arc_process);

            let push_back = match *front_arc_process.status.read().unwrap() {
                Status::Runnable => true,
                Status::Running => unreachable!(),
                Status::Exiting(ref _exception) => {
                    // TODO exit logging and linking
                    false
                }
            };

            let front_pid = front_arc_process.pid;

            if push_back {
                self.0.push_back(front_arc_process);
            }

            if front_pid == pid {
                break;
            }
        }
    }
}

pub struct Scheduler {
    pub id: ID,
    pub hierarchy: RwLock<Hierarchy>,
    // References are always 64-bits even on 32-bit platforms
    reference_count: AtomicU64,
    #[allow(dead_code)]
    run_queue_by_priority: HashMap<Priority, Arc<Mutex<RunQueue>>>,
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

    #[cfg(test)]
    pub fn run_queue(&self, priority: Priority) -> Arc<Mutex<RunQueue>> {
        Arc::clone(self.run_queue_by_priority.get(&priority).unwrap())
    }

    #[cfg(test)]
    pub fn run_through(&self, arc_process: &Arc<Process>) {
        let mut locked_run_queue = self
            .run_queue_by_priority
            .get(&arc_process.priority)
            .unwrap()
            .lock()
            .unwrap();

        locked_run_queue.run_through(arc_process)
    }

    pub fn spawn(
        parent_process: &Process,
        module: Term,
        function: Term,
        arguments: Vec<Term>,
    ) -> Arc<Process> {
        let process = Process::spawn(parent_process, module, function, arguments);
        let parent_locked_option_weak_scheduler = parent_process.scheduler.lock().unwrap();

        let arc_scheduler = match *parent_locked_option_weak_scheduler {
            Some(ref weak_scheduler) => match weak_scheduler.upgrade() {
                Some(arc_scheduler) => arc_scheduler,
                None => panic!(
                    "Parent process ({:?}) Scheduler has been Dropped",
                    parent_process
                ),
            },
            None => panic!(
                "Parent process ({:?}) is not assigned to a Scheduler",
                parent_process
            ),
        };

        let mut locked_run_queue = arc_scheduler
            .run_queue_by_priority
            .get(&process.priority)
            .unwrap()
            .lock()
            .unwrap();

        {
            let mut locked_option_weak_scheduler = process.scheduler.lock().unwrap();
            *locked_option_weak_scheduler = Some(Arc::downgrade(&arc_scheduler));
        }

        let arc_process = Arc::new(process);

        locked_run_queue.push_back(Arc::clone(&arc_process));

        put_pid_to_process(&arc_process);

        arc_process
    }

    pub fn spawn_init(self: Arc<Scheduler>) -> Arc<Process> {
        let arc_process = Arc::new(Process::init());
        let scheduler_arc_process = Arc::clone(&arc_process);

        // `parent_process.scheduler.lock` has to be taken first to even get the run queue in
        // `spawn`, so copy that lock order here.
        let mut locked_option_weak_scheduler = scheduler_arc_process.scheduler.lock().unwrap();

        let ref_arc_mutex_run_queue = self
            .run_queue_by_priority
            .get(&arc_process.priority)
            .unwrap();
        let mut locked_run_queue = ref_arc_mutex_run_queue.lock().unwrap();

        *locked_option_weak_scheduler = Some(Arc::downgrade(&self));
        locked_run_queue.push_back(Arc::clone(&arc_process));

        put_pid_to_process(&arc_process);

        arc_process
    }

    // Private

    fn new(raw_id: id::Raw) -> Scheduler {
        let mut run_queue_by_priority = HashMap::with_capacity(4);
        run_queue_by_priority.insert(Priority::Low, Default::default());
        run_queue_by_priority.insert(Priority::Normal, Default::default());
        run_queue_by_priority.insert(Priority::High, Default::default());
        run_queue_by_priority.insert(Priority::Max, Default::default());

        Scheduler {
            id: ID::new(raw_id),
            hierarchy: Default::default(),
            reference_count: AtomicU64::new(0),
            run_queue_by_priority,
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
            panic!(
                "Scheduler already registered with ID ({:?}",
                arc_scheduler.id
            );
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
            use crate::otp::erlang;

            #[test]
            fn different_processes_have_different_pids() {
                let erlang = Term::str_to_atom("erlang", DoNotCare).unwrap();
                let exit = Term::str_to_atom("exit", DoNotCare).unwrap();
                let normal = Term::str_to_atom("normal", DoNotCare).unwrap();

                let first_process =
                    Scheduler::spawn(&process::local::test_init(), erlang, exit, vec![normal]);
                let second_process = Scheduler::spawn(&first_process, erlang, exit, vec![normal]);

                assert_ne!(
                    erlang::self_0(&first_process),
                    erlang::self_0(&second_process)
                );
            }
        }
    }
}
