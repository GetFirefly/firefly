#![allow(unused)]
mod run_queue;

use std::alloc::Layout;
use std::fmt::{self, Debug};
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use hashbrown::HashMap;

use anyhow::anyhow;
use lazy_static::lazy_static;

use log::info;

use liblumen_core::locks::{Mutex, RwLock};
use liblumen_core::util::thread_local::ThreadLocalCell;

use liblumen_alloc::atom;
use liblumen_alloc::erts::apply;
use liblumen_alloc::erts::process;
use liblumen_alloc::erts::process::{CalleeSavedRegisters, Priority, Process, Status};
use liblumen_alloc::erts::scheduler::id;
use liblumen_alloc::erts::term::prelude::{Atom, ReferenceNumber, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_rt_core as rt_core;
use lumen_rt_core::process::CURRENT_PROCESS;
use lumen_rt_core::timer::Hierarchy;

const MAX_REDUCTION_COUNT: u32 = 20;

// External thread locals owned by the generated code
extern "C" {
    #[thread_local]
    static mut CURRENT_REDUCTION_COUNT: u32;
}

thread_local! {
  static SCHEDULER: Arc<Scheduler> = Scheduler::registered();
}

lazy_static! {
    static ref SCHEDULERS: Mutex<HashMap<id::ID, Weak<Scheduler>>> = Mutex::new(Default::default());
}

#[export_name = "__scheduler_stop_waiting"]
pub fn scheduler_stop_waiting(process: &Process) {
    let id = process.scheduler_id().unwrap();
    if let Some(scheduler) = SCHEDULERS.lock().get(&id).and_then(|s| s.upgrade()) {
        scheduler.stop_waiting(process)
    }
}

#[derive(Copy, Clone)]
struct StackPointer(*mut u64);

#[export_name = "__lumen_builtin_spawn"]
pub extern "C" fn builtin_spawn(to: Term, msg: Term) -> Term {
    unimplemented!()
}

#[export_name = "__lumen_builtin_yield"]
pub unsafe extern "C" fn process_yield() -> bool {
    let s = <Scheduler as rt_core::Scheduler>::current();
    // NOTE: We always set root=false here because the root
    // process never invokes this function
    s.process_yield(/* root= */ false)
}

#[naked]
#[inline(never)]
#[cfg(all(unix, target_arch = "x86_64"))]
#[export_name = "__lumen_builtin_return"]
pub unsafe extern "C" fn process_return_continuation() {
    let f: fn() -> () = process_return;
    asm!("
        callq *$0
        retq
        "
    :
    : "r"(f)
    :
    : "volatile", "alignstack"
    );
}

#[inline(never)]
fn process_return() {
    let s = <Scheduler as rt_core::Scheduler>::current();
    do_process_return(&s);
}

#[export_name = "__lumen_builtin_malloc"]
pub unsafe extern "C" fn builtin_malloc(kind: u32, arity: usize) -> *mut u8 {
    use core::convert::TryInto;
    use liblumen_core::alloc::Layout;
    use liblumen_term::TermKind;
    use liblumen_alloc::erts::term::prelude::*;
    use liblumen_alloc::erts::term::closure::ClosureLayout;

    let kind_result: Result<TermKind, _> = kind.try_into();
    match kind_result {
        Ok(TermKind::Closure) => {
            let s = <Scheduler as rt_core::Scheduler>::current();
            let cl = ClosureLayout::for_env_len(arity);
            let result = s.current.alloc_nofrag_layout(cl.layout().clone());
            if let Ok(nn) = result {
                return nn.as_ptr() as *mut u8;
            }
        }
        Ok(TermKind::Tuple) => {
            let s = <Scheduler as rt_core::Scheduler>::current();
            let layout = Tuple::layout_for_len(arity);
            let result = s.current.alloc_nofrag_layout(layout);
            if let Ok(nn) = result {
                return nn.as_ptr() as *mut u8;
            }
        }
        Ok(TermKind::Cons) => {
            let s = <Scheduler as rt_core::Scheduler>::current();
            let layout = Layout::new::<Cons>();
            let result = s.current.alloc_nofrag_layout(layout);
            if let Ok(nn) = result {
                return nn.as_ptr() as *mut u8;
            }
        }
        Ok(tk) => {
            unimplemented!("unhandled use of malloc for {:?}", tk);
        }
        Err(_) => {
            panic!("invalid term kind: {}", kind);
        }
    }

    ptr::null_mut()
}

/// Called when the current process has finished executing, and has
/// returned all the way to its entry function. This marks the process
/// as exiting (if it wasn't already), and then yields to the scheduler
fn do_process_return(scheduler: &Scheduler) -> bool {
    use liblumen_alloc::erts::term::prelude::*;
    if scheduler.current.pid() != scheduler.root.pid() {
        scheduler
            .current
            .exit(atom!("normal"), anyhow!("Out of code").into());
        // NOTE: We always set root=false here, even though this can
        // be called from the root process, since returning from the
        // root process exits the scheduler loop anyway, so no stack
        // swapping can occur
        scheduler.process_yield(/* root= */ false)
    } else {
        true
    }
}

pub struct Scheduler {
    id: id::ID,
    hierarchy: RwLock<Hierarchy>,
    // References are always 64-bits even on 32-bit platforms
    reference_count: AtomicU64,
    run_queues: RwLock<run_queue::Queues>,
    // Non-monotonic unique integers are scoped to the scheduler ID and then use this per-scheduler
    // `u64`.
    unique_integer: AtomicU64,
    root: Arc<Process>,
    init: ThreadLocalCell<Arc<Process>>,
    current: ThreadLocalCell<Arc<Process>>,
}
// This guarantee holds as long as `init` and `current` are only
// ever accessed by the scheduler when scheduling
unsafe impl Sync for Scheduler {}
impl rt_core::Scheduler for Scheduler {
    #[inline]
    fn current() -> Arc<Self> {
        SCHEDULER.with(|s| s.clone())
    }

    fn id(&self) -> id::ID {
        self.id
    }

    fn hierarchy(&self) -> &RwLock<Hierarchy> {
        &self.hierarchy
    }

    /// Gets the next available reference number
    fn next_reference_number(&self) -> ReferenceNumber {
        self.reference_count.fetch_add(1, Ordering::SeqCst)
    }
}
impl Scheduler {
    /// Creates a new scheduler with the default configuration
    fn new() -> anyhow::Result<Scheduler> {
        let id = id::next();

        // The root process is how the scheduler gets time for itself,
        // and is also how we know when to shutdown the scheduler due
        // to termination of all its processes
        let root = Arc::new(Process::new(
            Priority::Normal,
            None,
            Arc::new(ModuleFunctionArity {
                module: Atom::from_str("root"),
                function: Atom::from_str("init"),
                arity: 0,
            }),
            ptr::null_mut(),
            0,
        ));
        let run_queues = Default::default();
        Scheduler::spawn_root(root.clone(), id, &run_queues)?;

        // Placeholder
        let init = Arc::new(Process::new(
            Priority::Normal,
            None,
            Arc::new(ModuleFunctionArity {
                module: Atom::from_str("undef"),
                function: Atom::from_str("undef"),
                arity: 0,
            }),
            ptr::null_mut(),
            0,
        ));

        // The scheduler starts with the root process running
        let current = ThreadLocalCell::new(root.clone());

        Ok(Self {
            id,
            run_queues,
            root,
            init: ThreadLocalCell::new(init),
            current,
            hierarchy: Default::default(),
            reference_count: AtomicU64::new(0),
            unique_integer: AtomicU64::new(0),
        })
    }

    // Spawns the init process, should be called immediately after
    // scheduler creation
    pub fn init(&self) -> anyhow::Result<()> {
        // The init process is the actual "root" Erlang process, it acts
        // as the entry point for the program from Erlang's perspective,
        // and is responsible for starting/stopping the system in Erlang.
        //
        // If this process exits, the scheduler terminates
        let (init_heap, init_heap_size) = process::alloc::default_heap()?;
        let init = Arc::new(Process::new_with_stack(
            Priority::Normal,
            None,
            Arc::new(ModuleFunctionArity {
                module: Atom::from_str("init"),
                function: Atom::from_str("start"),
                arity: 0,
            }),
            init_heap,
            init_heap_size,
        )?);
        let clone = init.clone();
        unsafe { self.init.set(init); }
        Scheduler::spawn_internal(clone, self.id, &self.run_queues);

        Ok(())
    }

    /// Gets the scheduler registered to this thread
    ///
    /// If no scheduler has been created for this thread, one is created
    fn registered() -> Arc<Self> {
        let mut schedulers = SCHEDULERS.lock();
        let s = Arc::new(Self::new().unwrap());

        if let Some(_) = schedulers.insert(s.id, Arc::downgrade(&s)) {
            panic!("Scheduler already registered with ID ({:?}", s.id);
        }

        s
    }

    /// Gets a scheduler by its ID
    pub fn from_id(id: &id::ID) -> Option<Arc<Self>> {
        Self::current_from_id(id).or_else(|| SCHEDULERS.lock().get(id).and_then(|s| s.upgrade()))
    }

    /// Returns the current thread's scheduler if it matches the given ID
    fn current_from_id(id: &id::ID) -> Option<Arc<Self>> {
        SCHEDULER.with(|s| if &s.id == id { Some(s.clone()) } else { None })
    }

    /// Gets the next available unique integer
    pub fn next_unique_integer(&self) -> u64 {
        self.unique_integer.fetch_add(1, Ordering::SeqCst)
    }

    /// Returns the length of the current scheduler's run queue
    pub fn run_queues_len(&self) -> usize {
        self.run_queues.read().len()
    }

    /// Returns the length of a specific run queue in the current scheduler
    #[cfg(test)]
    pub fn run_queue_len(&self, priority: Priority) -> usize {
        self.run_queues.read().run_queue_len(priority)
    }

    /// Returns true if the given process is in the current scheduler's run queue
    #[cfg(test)]
    pub fn is_run_queued(&self, value: &Arc<Process>) -> bool {
        self.run_queues.read().contains(value)
    }

    pub fn stop_waiting(&self, process: &Process) {
        self.run_queues.write().stop_waiting(process);
    }

    // TODO: Request application master termination for controlled shutdown
    // This request will always come from the thread which spawned the application
    // master, i.e. the "main" scheduler thread
    //
    // Returns `Ok(())` if shutdown was successful, `Err(anyhow::Error)` if something
    // went wrong during shutdown, and it was not able to complete normally
    pub fn shutdown(&self) -> anyhow::Result<()> {
        // For now just Ok(()), but this needs to be addressed when proper
        // system startup/shutdown is in place
        CURRENT_PROCESS.with(|cp| cp.replace(None));
        Ok(())
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
        let mut locked_scheduler_by_id = SCHEDULERS.lock();

        locked_scheduler_by_id
            .remove(&self.id)
            .expect("Scheduler not registered");
    }
}
impl PartialEq for Scheduler {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

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
    ///
    /// Returns `true` if a process was run.  Returns `false` if no process could be run and the
    /// scheduler should sleep or work steal.
    #[must_use]
    pub fn run_once(&self) -> bool {
        // We always set root=true here, since calling this function is always done
        // from the scheduler loop, and only ever from the root context
        self.process_yield(/* root= */ true)
    }

    /// This function performs two roles, albeit virtually identical:
    ///
    /// First, this function is called by the scheduler to resume execution
    /// of a process pulled from the run queue. It does so using its "root"
    /// process as its context.
    ///
    /// Second, this function is called by a process when it chooses to
    /// yield back to the scheduler. In this case, the scheduler "root"
    /// process is swapped in, so the scheduler has a chance to do its
    /// auxilary tasks, after which the scheduler will call it again to
    /// swap in a new process.
    fn process_yield(&self, is_root: bool) -> bool {
        info!("entering core scheduler loop");
        self.hierarchy.write().timeout();

        loop {
            let next = {
                let mut rq = self.run_queues.write();
                rq.dequeue()
            };

            match next {
                Run::Now(process) => {
                    info!("found process to schedule");
                    // Don't allow exiting processes to run again.
                    //
                    // Without this check, a process.exit() from outside the process during WAITING
                    // will return to the Frame that called `process.wait()`
                    if !process.is_exiting() {
                        info!("swapping into process (is_root = {})", is_root);
                        unsafe {
                            self.swap_process(process, is_root);
                        }
                    } else {
                        info!("process is exiting");
                        process.reduce()
                    }

                    info!("exiting scheduler loop");
                    // When reached, either the process scheduled is the root process,
                    // or the process is exiting and we called .reduce(); either way we're
                    // returning to the main scheduler loop to check for signals, etc.
                    break true;
                }
                Run::Delayed => {
                    info!("found process, but it is delayed");
                    continue;
                }
                Run::None if is_root => {
                    info!("no processes remaining to schedule, exiting loop");
                    // If no processes are available, then the scheduler should steal,
                    // but if it can't/doesn't, then it must terminate, as there is
                    // nothing we can swap to. When we break here, we're returning
                    // to the core scheduler loop, which _must_ terminate, if it does
                    // not, we'll just end up right back here again.
                    //
                    // TODO: stealing
                    break false;
                }
                Run::None => unreachable!(),
            }
        }
    }

    /// This function takes care of coordinating the scheduling of a new
    /// process/descheduling of the current process.
    ///
    /// - Updating process status
    /// - Updating reduction count based on accumulated reductions during execution
    /// - Resetting reduction counter for next process
    /// - Handling exiting processes (logging/propagating)
    ///
    /// Once that is complete, it swaps to the new process stack via `swap_stack`,
    /// at which point execution resumes where the newly scheduled process left
    /// off previously, or in its init function.
    unsafe fn swap_process(&self, new: Arc<Process>, is_root: bool) {
        // Mark the new process as Running
        let new_ctx = &new.registers as *const _;
        {
            let mut new_status = new.status.write();
            *new_status = Status::Running;
        }

        // Replace the previous process with the new as the currently scheduled process
        let _ = CURRENT_PROCESS.with(|cp| cp.replace(Some(new.clone())));
        let prev = self.current.replace(new.clone());

        // Increment reduction count if not the root process
        if !is_root {
            let prev_reductions = reset_reduction_counter();
            prev.total_reductions
                .fetch_add(prev_reductions as u64, Ordering::Relaxed);
        }

        // Change the previous process status to Runnable
        {
            let mut prev_status = prev.status.write();
            if Status::Running == *prev_status {
                *prev_status = Status::Runnable
            }
        }

        // Save the previous process registers for the stack swap
        let prev_ctx = &prev.registers as *const _ as *mut _;

        // Then try to schedule it for the future
        // If the process is exiting, then handle the exit, otherwise
        // proceed to the stack swap
        if let Some(exiting) = self.run_queues.write().requeue(prev) {
            if let Status::Exiting(ref ex) = *exiting.status.read() {
                crate::process::log_exit(&exiting, ex);
                crate::process::propagate_exit(&exiting, ex);
            } else {
                unreachable!()
            }
        }

        // Execute the swap
        //
        // When swapping to the root process, we return here, which
        // will unwind back to the main scheduler loop in `lib.rs`.
        //
        // When swapping to a newly spawned process, we return "into"
        // its init function, or put another way, we jump to its
        // function prologue. In this situation, all of the saved registers
        // except %rsp and %rbp will be zeroed. %rsp is set during the call
        // to `spawn`, but %rbp is set to the current %rbp value to ensure
        // that stack traces link the new stack to the frame in which execution
        // started
        //
        // When swapping to a previously spawned process, we return here,
        // since the process called `process_yield`. From here we unwind back
        // to the call to `process_yield` and resume execution from the point
        // where it was called.
        swap_stack(prev_ctx, new_ctx);
    }

    /// Schedules the given process for execution
    pub fn schedule(&mut self, process: Arc<Process>) {
        debug_assert_ne!(
            Some(self.id),
            process.scheduler_id(),
            "process is already scheduled here!"
        );

        process.schedule_with(self.id);

        let mut rq = self.run_queues.write();
        rq.enqueue(process);
    }

    /// Spawns a new process using the given init function as its entry
    #[inline]
    pub fn spawn(&mut self, process: Arc<Process>) -> anyhow::Result<()> {
        Self::spawn_internal(process, self.id, &self.run_queues);
        Ok(())
    }

    // Root process uses the original thread stack, no initialization required.
    //
    // It also starts "running", so we don't put it on the run queue
    fn spawn_root(
        process: Arc<Process>,
        id: id::ID,
        _run_queues: &RwLock<run_queue::Queues>,
    ) -> anyhow::Result<()> {
        process.schedule_with(id);

        *process.status.write() = Status::Running;

        Ok(())
    }

    fn spawn_internal(process: Arc<Process>, id: id::ID, run_queues: &RwLock<run_queue::Queues>) {
        process.schedule_with(id);

        let mfa = &process.initial_module_function_arity;
        let init_fn_result = apply::find_symbol(&mfa);
        if init_fn_result.is_none() {
            panic!(
                "invalid mfa provided for process ({}), no such symbol found",
                &mfa
            );
        }
        let init_fn = init_fn_result.unwrap();

        #[inline(always)]
        unsafe fn push(sp: &mut StackPointer, value: u64) {
            sp.0 = sp.0.offset(-1);
            ptr::write(sp.0, value);
        }

        // Write the return function and init function to the end of the stack,
        // when execution resumes, the pointer before the stack pointer will be
        // used as the return address - the first time that will be the init function.
        //
        // When execution returns from the init function, then it will return via
        // `process_return`, which will return to the scheduler and indicate that
        // the process exited. The nature of the exit is indicated by error state
        // in the process itself
        unsafe {
            let mut sp = StackPointer(process.stack.top as *mut u64);
            // Function that will be called when returning from init_fn
            push(&mut sp, process_return_continuation as u64);
            // Function that the newly spawned process should call first
            push(&mut sp, init_fn as u64);
            // Update process stack pointer
            let s_top = &process.stack.top as *const _ as *mut _;
            ptr::write(s_top, sp.0 as *const u8);
            // Update rsp
            let rsp = &process.registers.rsp as *const u64 as *mut _;
            ptr::write(rsp, sp.0 as u64);
            let rbp = &process.registers.rbp as *const u64 as *mut _;
            ptr::write(rbp, sp.0 as u64);
        }

        *process.status.write() = Status::Runnable;

        let mut rq = run_queues.write();
        rq.enqueue(process);
    }
}

fn reset_reduction_counter() -> u64 {
    let count = unsafe { CURRENT_REDUCTION_COUNT };
    unsafe {
        CURRENT_REDUCTION_COUNT = 0;
    }
    count as u64
    //CURRENT_REDUCTION_COUNT.swap(0, Ordering::Relaxed)
}

/// This function uses inline assembly to save the callee-saved registers for the outgoing
/// process, and restore them for the incoming process. When this function returns, it will
/// resume execution where `swap_stack` was called previously.
#[naked]
#[inline(never)]
#[cfg(all(unix, target_arch = "x86_64"))]
unsafe fn swap_stack(prev: *mut CalleeSavedRegisters, new: *const CalleeSavedRegisters) {
    asm!("
        # Save the stack pointer, and callee-saved registers of `prev`
        movq     %rsp, ($0)
        movq     %r15, 8($0)
        movq     %r14, 16($0)
        movq     %r13, 24($0)
        movq     %r12, 32($0)
        movq     %rbx, 40($0)
        movq     %rbp, 48($0)

        # Restore the stack pointer, and callee-saved registers of `new`
        movq     ($1),   %rsp
        movq     8($1),  %r15
        movq     16($1), %r14
        movq     24($1), %r13
        movq     32($1), %r12
        movq     40($1), %rbx
        movq     48($1), %rbp

        # Return into the new process
        retq
        "
    :
    : "r"(prev), "r"(new)
    :
    : "volatile", "alignstack"
    );
}

#[cfg(not(all(unix, target_arch = "x86_64")))]
compile_error!("lumen_rt_minimal does not currently support this architecture!");
