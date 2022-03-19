use std::any::Any;
use std::convert::TryInto;
use std::ffi::c_void;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use liblumen_core::locks::RwLock;

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::exception::SystemException;
use liblumen_alloc::erts::process::{Frame, FrameWithArguments, Native, Priority, Process, Status};
pub use liblumen_alloc::erts::scheduler::{id, ID};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity, Ran};

use lumen_rt_core::process::spawn::options::Options;
use lumen_rt_core::process::{log_exit, propagate_exit, CURRENT_PROCESS};
use lumen_rt_core::registry::put_pid_to_process;
pub use lumen_rt_core::scheduler::{
    current, from_id, run_through, Scheduled, SchedulerDependentAlloc, Spawned,
};
use lumen_rt_core::scheduler::{run_queue, unregister, Run, Scheduler as SchedulerTrait};
use lumen_rt_core::timer::Hierarchy;

use crate::process::out_of_code;

// External functions defined in OTP
extern "C-unwind" {
    #[link_name = "erlang:apply/2"]
    fn apply_2(module: Term, function: Term, arguments: Term) -> Term;

    #[link_name = "erlang:apply/3"]
    fn apply_3(module: Term, function: Term, arguments: Term) -> Term;
}

#[export_name = "lumen_rt_scheduler_unregistered"]
fn unregistered() -> Arc<dyn lumen_rt_core::scheduler::Scheduler> {
    Arc::new(Scheduler {
        id: id::next(),
        hierarchy: Default::default(),
        reference_count: AtomicU64::new(0),
        run_queues: Default::default(),
        unique_integer: AtomicU64::new(0),
    })
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

    fn runnable(process: &Process, frame_with_arguments: FrameWithArguments) {
        process.runnable(|| {
            process.queue_frame_with_arguments(frame_with_arguments);
            process.queue_frame_with_arguments(out_of_code::frame().with_arguments(false, &[]));
            process.stack_queued_frames_with_arguments();
        })
    }

    fn spawn_closure_frame_with_arguments(
        process: &Process,
        closure: Boxed<Closure>,
    ) -> FrameWithArguments {
        let module_function_arity = ModuleFunctionArity {
            module: Atom::from_str("erlang"),
            function: Atom::from_str("apply"),
            arity: 2,
        };
        // I wish these was a safer way to say to strip "if and only if unsafe"
        let native = unsafe { Native::from_ptr(apply_2 as *const c_void, 2) };
        let frame = Frame::new(module_function_arity, native);

        let process_closure = closure.clone_to_process(process);
        let process_arguments = Term::NIL;

        frame.with_arguments(false, &[process_closure, process_arguments])
    }

    fn spawn_module_function_arguments_frame_with_arguments(
        process: &Process,
        module: Atom,
        function: Atom,
        arguments: Vec<Term>,
    ) -> FrameWithArguments {
        let module_function_arity = ModuleFunctionArity {
            module: Atom::from_str("erlang"),
            function: Atom::from_str("apply"),
            arity: 3,
        };
        // I wish these was a safer way to say to strip "if and only if unsafe"
        let native = unsafe { Native::from_ptr(apply_3 as *const c_void, 3) };
        let frame = Frame::new(module_function_arity, native);

        let process_module = module.encode().unwrap();
        let process_function = function.encode().unwrap();
        let process_argument_vec: Vec<Term> = arguments
            .iter()
            .map(|arguments| arguments.clone_to_process(process))
            .collect();
        let process_arguments = process.list_from_slice(&process_argument_vec);

        frame.with_arguments(
            false,
            &[process_module, process_function, process_arguments],
        )
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
                            Ran::Waiting | Ran::Reduced | Ran::Exited | Ran::RuntimeException => (),
                            Ran::SystemException => {
                                let runnable = match &*arc_process.status.read() {
                                    Status::SystemException(system_exception) => {
                                        match system_exception {
                                            SystemException::Alloc(_) => {
                                                let mut roots = [];
                                                match arc_process.garbage_collect(0, &mut roots[..])
                                                {
                                                    Ok(reductions) => {
                                                        arc_process.total_reductions.fetch_add(
                                                            reductions.try_into().unwrap(),
                                                            Ordering::SeqCst,
                                                        );

                                                        // Clear the status for `requeue` on
                                                        // successful `garbage_collect`
                                                        true
                                                    }
                                                    Err(gc_err) => panic!(
                                                        "fatal garbage collection error: {:?}",
                                                        gc_err
                                                    ),
                                                }
                                            }
                                            err => panic!("system error: {}", err),
                                        }
                                    }
                                    _ => unreachable!(),
                                };

                                if runnable {
                                    // Have to set after `match` where `ReadGuard` is held
                                    *arc_process.status.write() = Status::Runnable;
                                }
                            }
                        }
                    } else {
                        arc_process.reduce()
                    }

                    // Don't `if let` or `match` on the return from `requeue` as it will keep the
                    // lock on the `run_queue`, causing a dead lock when `propagate_exit` calls
                    // `Scheduler::stop_waiting` for any linked or monitoring process.
                    let option_exiting_arc_process = self.run_queues.write().requeue(arc_process);

                    if let Some(exiting_arc_process) = option_exiting_arc_process {
                        match *exiting_arc_process.status.read() {
                            Status::Exited => {
                                propagate_exit(&exiting_arc_process, None);
                            }
                            Status::RuntimeException(ref exception) => {
                                log_exit(&exiting_arc_process, exception);
                                propagate_exit(&exiting_arc_process, Some(exception));
                            }
                            _ => unreachable!(),
                        }
                    }

                    CURRENT_PROCESS.with(|current_process| current_process.replace(None));

                    break true;
                }
                Run::Delayed => continue,
                Run::Waiting => break true,
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
        debug_assert_ne!(
            Some(self.id),
            process.scheduler_id(),
            "process is already scheduled here!"
        );
        assert_eq!(*process.status.read(), Status::Runnable);

        process.schedule_with(self.id);

        let arc_process = Arc::new(process);

        self.run_queues.write().enqueue(arc_process.clone());
        put_pid_to_process(&arc_process);

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

    fn spawn_init(&self, minimum_heap_size: usize) -> anyhow::Result<Arc<Process>> {
        let mut options: Options = Default::default();
        options.min_heap_size = Some(minimum_heap_size);

        let Spawned { arc_process, .. } = self.spawn_module_function_arguments(
            None,
            Atom::from_str("init"),
            Atom::from_str("start"),
            vec![],
            options,
        )?;

        Ok(arc_process)
    }

    fn spawn_closure(
        &self,
        parent: Option<&Process>,
        closure: Boxed<Closure>,
        options: Options,
    ) -> anyhow::Result<Spawned> {
        let (heap, heap_size) = options.sized_heap()?;
        let priority = options.cascaded_priority(parent);
        let initial_module_function_arity = closure.module_function_arity();
        let process = Process::new(
            priority,
            parent,
            initial_module_function_arity,
            heap,
            heap_size,
        );

        let frame_with_arguments = Self::spawn_closure_frame_with_arguments(&process, closure);
        Self::runnable(&process, frame_with_arguments);

        let connection = options.connect(parent, &process);

        let arc_process = match parent {
            Some(parent) => parent.scheduler().unwrap().schedule(process),
            None => self.schedule(process),
        };

        Ok(Spawned {
            arc_process,
            connection,
        })
    }

    fn spawn_module_function_arguments(
        &self,
        parent: Option<&Process>,
        module: Atom,
        function: Atom,
        arguments: Vec<Term>,
        options: Options,
    ) -> anyhow::Result<Spawned> {
        let (heap, heap_size) = options.sized_heap()?;
        let priority = options.cascaded_priority(parent);
        let initial_module_function_arity = ModuleFunctionArity {
            module,
            function,
            arity: arguments.len() as Arity,
        };
        let process = Process::new(
            priority,
            parent,
            initial_module_function_arity,
            heap,
            heap_size,
        );

        let frame_with_arguments = Self::spawn_module_function_arguments_frame_with_arguments(
            &process, module, function, arguments,
        );
        Self::runnable(&process, frame_with_arguments);

        let connection = options.connect(parent, &process);

        let arc_process = match parent {
            Some(parent) => parent.scheduler().unwrap().schedule(process),
            None => self.schedule(process),
        };

        Ok(Spawned {
            arc_process,
            connection,
        })
    }

    fn stop_waiting(&self, process: &Process) {
        process.stop_waiting();
        self.run_queues.write().stop_waiting(process);
    }
}
