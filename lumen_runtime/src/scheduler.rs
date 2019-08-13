use core::fmt::{self, Debug};
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::sync::{Arc, Weak};

use hashbrown::HashMap;

use liblumen_core::locks::{Mutex, RwLock};

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::Code;
#[cfg(test)]
use liblumen_alloc::erts::process::Priority;
use liblumen_alloc::erts::process::{ProcessControlBlock, Status};
pub use liblumen_alloc::erts::scheduler::{id, ID};
use liblumen_alloc::erts::term::{reference, Atom, Reference, Term, TypedTerm};

use crate::process;
use crate::registry::put_pid_to_process;
use crate::run;
use crate::run::Run;
use crate::system;
use crate::timer::Hierarchy;

pub trait Scheduled {
    fn scheduler(&self) -> Option<Arc<Scheduler>>;
}

impl Scheduled for ProcessControlBlock {
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
    fn run_once(&self) -> bool {
        match self.hierarchy.write().timeout() {
            Ok(()) => (),
            Err(_) => unimplemented!(
                "GC _everything_ to get free space for `Messages.alloc` during `send_message`"
            ),
        };

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
                        match ProcessControlBlock::run(&arc_process) {
                            Ok(()) => (),
                            Err(exception) => unimplemented!(
                                "{:?} {:?}\n{:?}",
                                arc_process,
                                exception,
                                *arc_process.acquire_heap()
                            ),
                        }
                    }

                    match self.run_queues.write().requeue(arc_process) {
                        Some(exiting_arc_process) => match *exiting_arc_process.status.read() {
                            Status::Exiting(ref exception) => match exception.class {
                                runtime::Class::Exit => {
                                    let reason = exception.reason;

                                    if !is_expected_exit_reason(reason) {
                                        system::io::puts(&format!(
                                            "** (EXIT from {}) exited with reason: {}",
                                            exiting_arc_process, reason
                                        ));
                                    }
                                }
                                runtime::Class::Error { .. } => {
                                    system::io::puts(
                                        &format!(
                                            "** (EXIT from {}) exited with reason: an exception was raised: {}\n{}",
                                            exiting_arc_process,
                                            exception.reason,
                                            exiting_arc_process.stacktrace()
                                        )
                                    )
                                }
                                _ => unimplemented!("{:?}", exception),
                            },
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

    /// Returns `true` if `arc_process` was run; otherwise, `false`.
    #[must_use]
    pub fn run_through(&self, arc_process: &Arc<ProcessControlBlock>) -> bool {
        let ordering = Ordering::SeqCst;
        let reductions_before = arc_process.total_reductions.load(ordering);

        // The same as `run`, but stops when the process is run once
        loop {
            if self.run_once() {
                if arc_process.total_reductions.load(Ordering::SeqCst) <= reductions_before {
                    break true;
                } else {
                    continue;
                }
            } else {
                break false;
            }
        }
    }

    fn schedule(
        self: Arc<Scheduler>,
        process_control_block: ProcessControlBlock,
    ) -> Arc<ProcessControlBlock> {
        let mut writable_run_queues = self.run_queues.write();

        process_control_block.schedule_with(self.id);

        let arc_process_control_block = Arc::new(process_control_block);

        writable_run_queues.enqueue(Arc::clone(&arc_process_control_block));

        arc_process_control_block
    }

    /// Spawns a process with arguments for `apply(module, function, arguments)` on its stack.
    ///
    /// This allows the `apply/3` code to be changed with `apply_3::set_code(code)` to handle new
    /// MFA unique to a given application.
    pub fn spawn_apply_3(
        parent_process: &ProcessControlBlock,
        module: Atom,
        function: Atom,
        arguments: Term,
        heap: *mut Term,
        heap_size: usize,
    ) -> Result<Arc<ProcessControlBlock>, Alloc> {
        let process =
            process::spawn_apply_3(parent_process, module, function, arguments, heap, heap_size)?;
        let arc_scheduler = parent_process.scheduler().unwrap();
        let arc_process = arc_scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        Ok(arc_process)
    }

    /// Spawns a process with `arguments` on its stack and `code` run with those arguments instead
    /// of passing through `apply/3`.
    pub fn spawn(
        parent_process: &ProcessControlBlock,
        module: Atom,
        function: Atom,
        arguments: Vec<Term>,
        code: Code,
        heap: *mut Term,
        heap_size: usize,
    ) -> Result<Arc<ProcessControlBlock>, Alloc> {
        let process = process::spawn(
            parent_process,
            module,
            function,
            arguments,
            code,
            heap,
            heap_size,
        )?;
        let arc_scheduler = parent_process.scheduler().unwrap();
        let arc_process = arc_scheduler.schedule(process);

        put_pid_to_process(&arc_process);

        Ok(arc_process)
    }

    pub fn spawn_init(
        self: Arc<Scheduler>,
        minimum_heap_size: usize,
    ) -> Result<Arc<ProcessControlBlock>, Alloc> {
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

    pub fn stop_waiting(&self, process: &ProcessControlBlock) {
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

thread_local! {
  static SCHEDULER: Arc<Scheduler> = Scheduler::registered();
}

lazy_static! {
    static ref SCHEDULER_BY_ID: Mutex<HashMap<ID, Weak<Scheduler>>> =
        Mutex::new(Default::default());
}

fn is_expected_exit_reason(reason: Term) -> bool {
    match reason.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => match atom.name() {
            "normal" | "shutdown" => true,
            _ => false,
        },
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::Tuple(tuple) => {
                tuple.len() == 2 && {
                    match tuple[0].to_typed_term().unwrap() {
                        TypedTerm::Atom(atom) => atom.name() == "shutdown",
                        _ => false,
                    }
                }
            }
            _ => false,
        },
        _ => false,
    }
}

#[cfg(test)]
pub fn with_process<F>(f: F)
where
    F: FnOnce(&ProcessControlBlock) -> (),
{
    f(&process::test(&process::test_init()))
}

#[cfg(test)]
pub fn with_process_arc<F>(f: F)
where
    F: FnOnce(Arc<ProcessControlBlock>) -> (),
{
    f(process::test(&process::test_init()))
}

#[cfg(test)]
mod tests {
    use super::*;

    mod scheduler {
        use super::*;

        mod new_process {
            use super::*;

            use liblumen_alloc::erts::process::default_heap;
            use liblumen_alloc::erts::term::atom_unchecked;

            use crate::code;
            use crate::otp::erlang;

            #[test]
            fn different_processes_have_different_pids() {
                let erlang = Atom::try_from_str("erlang").unwrap();
                let exit = Atom::try_from_str("exit").unwrap();
                let normal = atom_unchecked("normal");
                let parent_arc_process_control_block = process::test_init();

                let first_process_arguments = parent_arc_process_control_block
                    .list_from_slice(&[normal])
                    .unwrap();
                let (first_heap, first_heap_size) = default_heap().unwrap();
                let first_process = Scheduler::spawn_apply_3(
                    &parent_arc_process_control_block,
                    erlang,
                    exit,
                    first_process_arguments,
                    first_heap,
                    first_heap_size,
                )
                .unwrap();

                let second_process_arguments = parent_arc_process_control_block
                    .list_from_slice(&[normal])
                    .unwrap();
                let (second_heap, second_heap_size) = default_heap().unwrap();
                let second_process = Scheduler::spawn_apply_3(
                    &first_process,
                    erlang,
                    exit,
                    second_process_arguments,
                    second_heap,
                    second_heap_size,
                )
                .unwrap();

                assert_ne!(
                    erlang::self_0(&first_process),
                    erlang::self_0(&second_process)
                );
            }
        }
    }
}
