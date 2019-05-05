use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

#[cfg(test)]
use crate::process::local::put_pid_to_process;
use crate::process::Process;
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

pub struct Scheduler {
    pub id: ID,
    pub hierarchy: Mutex<Hierarchy>,
    pub process_by_pid: RwLock<HashMap<Term, Arc<Process>>>,
    // References are always 64-bits even on 32-bit platforms
    reference_count: AtomicU64,
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

    #[cfg(test)]
    pub fn new_process(&self) -> Arc<Process> {
        let process = Process::new();
        let process_arc = Arc::new(process);

        put_pid_to_process(process_arc.clone());

        process_arc
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

    // Private

    fn new(raw_id: id::Raw) -> Scheduler {
        Scheduler {
            id: ID::new(raw_id),
            hierarchy: Default::default(),
            process_by_pid: Default::default(),
            reference_count: AtomicU64::new(0),
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
    f(&Scheduler::current().new_process())
}

#[cfg(test)]
pub fn with_process_arc<F>(f: F)
where
    F: FnOnce(Arc<Process>) -> (),
{
    f(Scheduler::current().new_process())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod scheduler {
        use super::*;

        mod new_process {
            use super::*;

            use crate::otp::erlang;

            #[test]
            fn different_processes_have_different_pids() {
                let scheduler = Scheduler::current();
                let first_process = scheduler.new_process();
                let second_process = scheduler.new_process();

                assert_ne!(
                    erlang::self_0(&first_process),
                    erlang::self_0(&second_process)
                );
            }
        }
    }
}
