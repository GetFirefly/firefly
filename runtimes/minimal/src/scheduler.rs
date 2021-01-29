use std::alloc::Layout;
use std::any::Any;
use std::ffi::c_void;
use std::fmt::{self, Debug};
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use log::info;

use liblumen_core::locks::RwLock;
use liblumen_core::sys::dynamic_call::DynamicCallee;
use liblumen_core::util::thread_local::ThreadLocalCell;

use liblumen_alloc::erts::exception::ErlangException;
use liblumen_alloc::erts::process::{CalleeSavedRegisters, Priority, Process, Status};
use liblumen_alloc::erts::scheduler::{id, ID};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::{Arity, CloneToProcess};

use lumen_rt_core::process::spawn::options::Options;
use lumen_rt_core::process::{log_exit, propagate_exit, CURRENT_PROCESS};
use lumen_rt_core::registry::put_pid_to_process;
use lumen_rt_core::scheduler::Scheduler as SchedulerTrait;
use lumen_rt_core::scheduler::{self, run_queue, unregister, Run};
pub use lumen_rt_core::scheduler::{
    current, from_id, run_through, Scheduled, SchedulerDependentAlloc, Spawned,
};
use lumen_rt_core::timer::Hierarchy;

// External thread locals owned by the generated code
extern "C" {
    #[thread_local]
    static mut CURRENT_REDUCTION_COUNT: u32;

    #[unwind(allowed)]
    #[link_name = "__lumen_trap_exceptions"]
    fn trap_exceptions_impl() -> bool;
}

// External functions defined in OTP
extern "C" {
    #[unwind(allowed)]
    #[link_name = "lumen:apply_apply_2/1"]
    fn apply_apply_2() -> usize;

    #[unwind(allowed)]
    #[link_name = "lumen:apply_apply_3/1"]
    fn apply_apply_3() -> usize;
}

crate fn stop_waiting(process: &Process) {
    if let Some(scheduler) = from_id(&process.scheduler_id().unwrap()) {
        scheduler.stop_waiting(process)
    }
}

#[derive(Copy, Clone)]
struct StackPointer(*mut u64);

#[unwind(allowed)]
#[export_name = "__lumen_builtin_yield"]
pub unsafe extern "C" fn process_yield() -> bool {
    scheduler::current()
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap()
        .process_yield()
}

#[unwind(allowed)]
#[export_name = "__lumen_builtin_exit"]
pub unsafe extern "C" fn process_exit(exception: Option<NonNull<ErlangException>>) {
    let arc_dyn_scheduler = scheduler::current();
    let scheduler = arc_dyn_scheduler
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap();

    if let Some(nn) = exception {
        let mut exception: Box<ErlangException> = Box::from_raw(nn.as_ptr());
        if exception.kind() == Atom::str_to_term("throw") {
            // Need to update reason to {nocatch, Reason}
            exception.set_nocatch();
        }
        scheduler.current.erlang_exit(exception);
    } else {
        scheduler.current.exit_normal();
    }
    scheduler.process_yield();
}

#[unwind(allowed)]
#[export_name = "__lumen_builtin_malloc"]
pub unsafe extern "C" fn builtin_malloc(kind: u32, arity: usize) -> *mut u8 {
    use core::convert::TryInto;
    use liblumen_alloc::erts::term::closure::ClosureLayout;
    use liblumen_alloc::erts::term::prelude::*;
    use liblumen_term::TermKind;

    let arc_dyn_scheduler = scheduler::current();
    let s = arc_dyn_scheduler
        .as_any()
        .downcast_ref::<Scheduler>()
        .unwrap();
    let process = &s.current;
    let kind_result: Result<TermKind, _> = kind.try_into();
    let layout = match kind_result {
        Ok(TermKind::Closure) => ClosureLayout::for_env_len(arity).layout().clone(),
        Ok(TermKind::Tuple) => Tuple::layout_for_len(arity),
        Ok(TermKind::Cons) => Layout::new::<Cons>(),
        Ok(tk) => {
            unimplemented!("unhandled use of malloc for {:?}", tk);
        }
        Err(_) => {
            panic!("invalid term kind: {}", kind);
        }
    };

    let result = process
        .alloc_nofrag_layout(layout.clone())
        .or_else(|_| process.alloc_fragment_layout(layout));

    match result {
        Ok(nn) => nn.as_ptr() as *mut u8,
        Err(_) => ptr::null_mut(),
    }
}

#[unwind(allowed)]
#[export_name = "lumen_rt_scheduler_unregistered"]
fn unregistered() -> Arc<dyn lumen_rt_core::scheduler::Scheduler> {
    Arc::new(Scheduler::new().unwrap())
}

pub struct Scheduler {
    pub id: id::ID,
    pub hierarchy: RwLock<Hierarchy>,
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
            ModuleFunctionArity {
                module: Atom::from_str("root"),
                function: Atom::from_str("init"),
                arity: 0,
            },
            ptr::null_mut(),
            0,
        ));
        let run_queues = Default::default();
        Scheduler::spawn_root(root.clone(), id, &run_queues)?;

        // Placeholder
        let init = Arc::new(Process::new(
            Priority::Normal,
            None,
            ModuleFunctionArity {
                module: Atom::from_str("undef"),
                function: Atom::from_str("undef"),
                arity: 0,
            },
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

    /// Returns true if the given process is in the current scheduler's run queue
    #[cfg(test)]
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
    fn eq(&self, other: &Self) -> bool {
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
        // The scheduler will yield to a process to execute
        self.scheduler_yield()
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

    fn spawn_init(&self, minimum_heap_size: usize) -> anyhow::Result<Arc<Process>> {
        // The init process is the actual "root" Erlang process, it acts
        // as the entry point for the program from Erlang's perspective,
        // and is responsible for starting/stopping the system in Erlang.
        //
        // If this process exits, the scheduler terminates
        let mut options: Options = Default::default();
        options.min_heap_size = Some(minimum_heap_size);

        let Spawned { arc_process, .. } = self.spawn_module_function_arguments(
            None,
            Atom::from_str("init"),
            Atom::from_str("start"),
            vec![],
            options,
        )?;

        unsafe {
            self.init.set(arc_process.clone());
        }

        Ok(arc_process)
    }

    /// Spawns a new process from the given parent, using the given closure as its entry
    fn spawn_closure(
        &self,
        parent: Option<&Process>,
        closure: Boxed<Closure>,
        options: Options,
    ) -> anyhow::Result<Spawned> {
        let (heap, heap_size) = options.sized_heap()?;
        let priority = options.cascaded_priority(parent);
        let initial_module_function_arity = closure.module_function_arity();
        let process = Process::new_with_stack(
            priority,
            parent,
            initial_module_function_arity,
            heap,
            heap_size,
        )?;

        let (init_fn, env) = Self::spawn_closure_init_env(&process, closure);
        Self::runnable(&process, init_fn, env);

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
        let process = Process::new_with_stack(
            priority,
            parent,
            initial_module_function_arity,
            heap,
            heap_size,
        )?;
        let (init_fn, env) =
            Self::spawn_module_function_arguments_init_env(&process, module, function, arguments);
        Self::runnable(&process, init_fn, env);

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

    // TODO: Request application master termination for controlled shutdown
    // This request will always come from the thread which spawned the application
    // master, i.e. the "main" scheduler thread
    //
    // Returns `Ok(())` if shutdown was successful, `Err(anyhow::Error)` if something
    // went wrong during shutdown, and it was not able to complete normally
    fn shutdown(&self) -> anyhow::Result<()> {
        // For now just Ok(()), but this needs to be addressed when proper
        // system startup/shutdown is in place
        CURRENT_PROCESS.with(|cp| cp.replace(None));
        Ok(())
    }

    fn stop_waiting(&self, process: &Process) {
        process.stop_waiting();
        self.run_queues.write().stop_waiting(process);
    }
}

impl Scheduler {
    fn process_yield(&self) -> bool {
        // Swap back to the scheduler, which will look like
        // a return from `swap_stack`. This function will
        // appear to return if the process that yielded is
        // rescheduled

        let scheduler_ctx = &self.root.registers as *const _ as *mut _;
        let process_ctx = &self.current.registers as *const _ as *mut _;
        unsafe {
            swap_stack(process_ctx, scheduler_ctx);
        }
        true
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
    fn scheduler_yield(&self) -> bool {
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
                    // will return to code that called `process.wait()`
                    let requeue_arc_process = if !process.is_exiting() {
                        info!("swapping into process {:?}", process.pid());
                        // The swap takes care of setting up the to-be-scheduled process
                        // as the current process, and swaps to its stack. The code below
                        // is executed when that process has yielded and we're resetting
                        // the state of the scheduler such that the "current process" is
                        // the scheduler itself
                        unsafe {
                            self.swap_process(process);
                        }

                        // When we reach here, the process has yielded
                        // back to the scheduler, and is still marked
                        // as the current process. We need to handle
                        // swapping it out with the scheduler process
                        // and handling its exit, if exiting
                        let _ = CURRENT_PROCESS.with(|cp| cp.replace(Some(self.root.clone())));
                        let prev = unsafe { self.current.replace(self.root.clone()) };

                        // Increment reduction count if not the root process
                        let prev_reductions = reset_reduction_counter();
                        prev.total_reductions
                            .fetch_add(prev_reductions as u64, Ordering::Relaxed);

                        // Change the previous process status to Runnable
                        {
                            let mut prev_status = prev.status.write();
                            if Status::Running == *prev_status {
                                *prev_status = Status::Runnable
                            }
                        }

                        prev
                    } else {
                        info!("process is exiting");
                        process.reduce();

                        process
                    };

                    // Try to schedule it for the future
                    //
                    // Don't `if let` or `match` on the return from `requeue` as it will keep the
                    // lock on the `run_queue`, causing a dead lock when `propagate_exit` calls
                    // `Scheduler::stop_waiting` for any linked or monitoring process.
                    let option_exiting_arc_process =
                        self.run_queues.write().requeue(requeue_arc_process);

                    // If the process is exiting, then handle the exit
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

                    info!("exiting scheduler loop after run");
                    // When reached, either the process scheduled is the root process,
                    // or the process is exiting and we called .reduce(); either way we're
                    // returning to the main scheduler loop to check for signals, etc.
                    break true;
                }
                Run::Delayed => {
                    info!("found process, but it is delayed");
                    continue;
                }
                Run::Waiting => {
                    info!("exiting scheduler loop because waiting");
                    // Return to main scheduler loop to check for signals and to re-enter from
                    // `run_once` and increment timeouts to knock out of waiting.
                    break true;
                }
                Run::None if self.current.pid() == self.root.pid() => {
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
    unsafe fn swap_process(&self, new: Arc<Process>) {
        // Mark the new process as Running
        let new_ctx = &new.registers as *const _;
        {
            let mut new_status = new.status.write();
            *new_status = Status::Running;
        }

        // Replace the previous process with the new as the currently scheduled process
        let _ = CURRENT_PROCESS.with(|cp| cp.replace(Some(new.clone())));
        let prev = self.current.replace(new.clone());

        // Change the previous process status to Runnable
        {
            let mut prev_status = prev.status.write();
            if Status::Running == *prev_status {
                *prev_status = Status::Runnable
            }
        }

        // Save the previous process registers for the stack swap
        let prev_ctx = &prev.registers as *const _ as *mut _;

        // Execute the swap
        //
        // When swapping to the root process, we effectively return from here, which
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
        // When swapping to a previously spawned process, we return to the end
        // of `process_yield`, which is what the process last called before the
        // scheduler was swapped in.
        swap_stack(prev_ctx, new_ctx);
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

        let r13 = &process.registers.r13 as *const u64 as *mut _;
        unsafe {
            ptr::write(r13, 0x0 as u64);
        }

        Ok(())
    }

    fn spawn_closure_init_env(
        process: &Process,
        closure: Boxed<Closure>,
    ) -> (DynamicCallee, Option<Term>) {
        let init_fn = unsafe { mem::transmute::<_, DynamicCallee>(apply_apply_2 as *const c_void) };
        let function = closure.clone_to_process(process);
        let arguments = Term::NIL;
        let env = Some(process.list_from_slice(&[function, arguments]));

        (init_fn, env)
    }

    fn spawn_module_function_arguments_init_env(
        process: &Process,
        module: Atom,
        function: Atom,
        arguments: Vec<Term>,
    ) -> (DynamicCallee, Option<Term>) {
        let init_fn = unsafe { mem::transmute::<_, DynamicCallee>(apply_apply_3 as *const c_void) };

        let process_module = module.encode().unwrap();
        let process_function = function.encode().unwrap();
        let process_argument_vec: Vec<Term> = arguments
            .iter()
            .map(|argument| argument.clone_to_process(process))
            .collect();
        let process_arguments = process.list_from_slice(&process_argument_vec);
        let env =
            Some(process.list_from_slice(&[process_module, process_function, process_arguments]));

        (init_fn, env)
    }

    fn runnable(process: &Process, init_fn: DynamicCallee, env: Option<Term>) {
        process.runnable(|| {
            #[allow(unused)]
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
                let stack = process.stack.lock();
                // This can be used to push items on the process
                // stack before it starts executing. For now that
                // is not being done
                let sp = StackPointer(stack.top as *mut u64);

                // Update process stack pointer
                let s_top = &stack.top as *const _ as *mut _;
                ptr::write(s_top, sp.0 as *const u8);

                // Write %rsp/%rbp initial values
                let rsp = &process.registers.rsp as *const u64 as *mut _;
                let rbp = &process.registers.rbp as *const u64 as *mut _;
                ptr::write(rsp, sp.0 as u64);
                ptr::write(rbp, sp.0 as u64);

                // If this init function has a closure env, place it in
                // r12, which will be moved to %rsi by swap_stack, such
                // that it becomes the second argument to __lumen_trap_exceptions,
                // which will in turn move it to %rdi to become the first
                // argument to the start function of the process
                let r12 = &process.registers.r12 as *const _ as *mut Term;
                ptr::write(r12, env.unwrap_or(Term::NONE));

                // This is used to indicate to swap_stack that this process
                // is being swapped to for the first time, which allows the
                // function to perform some initial one-time setup to link
                // call frames for the unwinder and call __lumen_trap_exceptions
                let r13 = &process.registers.r13 as *const u64 as *mut _;
                ptr::write(r13, FIRST_SWAP);

                // The function that __lumen_trap_exceptions will call as entry
                let r14 = &process.registers.r14 as *const u64 as *mut _;
                ptr::write(r14, init_fn as u64);

                // The function that swap_stack will call as entry (__lumen_trap_exceptions)
                let r15 = &process.registers.r15 as *const u64 as *mut _;
                ptr::write(r15, trap_exceptions_impl as u64);
            }
        })
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

/// `naked` functions can't take parameters because `naked` means it has a calling convention
/// that Rust can't understand, so they need a wrapper function (this function) to call the
/// naked function in asm
#[inline(never)]
#[unwind(allowed)]
#[cfg(all(unix, target_arch = "x86_64"))]
unsafe extern "C" fn swap_stack(prev: *mut CalleeSavedRegisters, new: *const CalleeSavedRegisters) {
    asm!(
      "call {}",
      sym naked_swap_stack,
      in("rdi") prev, in("rsi") new, in("rdx") FIRST_SWAP
    )
}

const FIRST_SWAP: u64 = 0xdeadbeef;

/// This function uses inline assembly to save the callee-saved registers for the outgoing
/// process, and restore them for the incoming process. When this function returns, it will
/// resume execution where `swap_stack` was called previously.
#[naked]
#[inline(never)]
#[unwind(allowed)]
#[cfg(all(unix, target_arch = "x86_64"))]
unsafe extern "C" fn naked_swap_stack() {
    asm!(
        // Save the return address to a register
        "lea rax, [rip+0f]",
        // Save the parent base pointer for when control returns to this call frame.
        // CFA directives will inform the unwinder to expect %rbp at the bottom of the
        // stack for this frame, so this should be the last value on the stack in the caller
        "push rbp",
        // We also save %rbp and %rsp to registers so that we can setup CFA directives if this
        // is the first swap for the target process
        "mov rcx, rbp",
        "mov r9,  rsp",
        // Save the stack pointer, and callee-saved registers of `prev` set in `rdi` by
        // `swap_stack` above
        "mov [rdi],    rsp",
        "mov [rdi+8],  r15",
        "mov [rdi+16], r14",
        "mov [rdi+24], r13",
        "mov [rdi+32], r12",
        "mov [rdi+40], rbx",
        "mov [rdi+48], rbp",
        // Restore the stack pointer, and callee-saved registers of `new` set in `rsi` by
        // `swap_stack` above
        "mov rsp, [rsi]",
        "mov r15, [rsi+8]",
        "mov r14, [rsi+16]",
        "mov r13, [rsi+24]",
        "mov r12, [rsi+32]",
        "mov rbx, [rsi+40]",
        "mov rbp, [rsi+48]",
        // The value of all the callee-saved registers has changed, so we
        // need to inform the unwinder of that fact before proceeding
        ".cfi_restore rsp",
        ".cfi_restore r15",
        ".cfi_restore r14",
        ".cfi_restore r13",
        ".cfi_restore r12",
        ".cfi_restore rbx",
        ".cfi_restore rbp",
        // If this is the first time swapping to this process,
        // we need to to perform some one-time initialization to
        // link the stack to the original parent stack (i.e. the scheduler),
        // which is important for the unwinder
        //
        // cmp doesn't support 64-bit immediates, so the constant is put in `rdx` in `swap_stack`
        // instead.
        "cmp rdx, r13",
        "jne .L1",
        // Ensure we never perform initialization twice
        "mov  r13, 0x0",
        // Store the original base pointer at the top of the stack
        "push rcx",
        // Followed by the return address
        "push rax",
        // Finally we store a pointer to the bottom of the stack in the
        // parent call frame. The unwinder will expect to restore %rbp
        // from this address
        "push r9",
        // These CFI directives inform the unwinder of where it can expect
        // to find the CFA relative to %rbp. This matches how we've laid out the stack.
        //
        // - The current %rbp is now 24 bytes (3 words) above %rsp.
        // - 16 bytes _down_ from the current %rbp is the value from %r9 that
        // we pushed, containing the parent call frame's stack pointer.
        //
        // The first directive tells the unwinder that it can expect to find the
        // CFA (call frame address) 16 bytes above %rbp. The second directive then
        // tells the unwinder that it can find the previous %rbp 16 bytes _down_
        // from the current %rbp. The result is that the unwinder will restore %rbp
        // from that stack slot, and will then expect to find the previous CFA 16 bytes
        // above that address, allowing the unwinder to walk back into the parent frame
        ".cfi_def_cfa rbp, 16",
        ".cfi_offset rbp, -16",
        // Now that the frames are linked, we can call the entry point. For now, this
        // is __lumen_trap_exceptions, which expects to receive two arguments: the function
        // being wrapped by the exception handler, and the value of the closure environment,
        // _if_ it is a closure being called, otherwise the value of that argument is Term::NONE
        "mov rdi, r14",
        "mov rsi, r12",
        // We have already set up the stack precisely, so we don't use callq here, instead
        // we go ahead and jump straight to the beginning of the entry function.
        // NOTE: This call never truly returns, as the exception handler calls __lumen_builtin_exit
        // with the return value of the 'real' entry function, or with an exception if one
        // is caught. However, swap_stack _does_ return for all other swaps, just not the first.
        "jmp r15",
        "1:",
        // We land here only on a context switch, and since the last switch _away_ from
        // this process pushed %rbp on to the stack, and we don't need that value, we
        // adjust the stack pointer accordingly.
        "add rsp, 8",
        // At this point we will return back to where execution left off:
        // For the 'root' (scheduler) process, this returns back into `swap_process`;
        // for all other processes, this returns to the code which was executing when
        // it yielded, the address of which is 8 bytes above the current stack pointer.
        // We pop and jmp rather than ret to avoid branch mispredictions.
        "pop rax",
        "jmp rax",
        "0:",
        // asm in naked functions must use `noreturn` option
        options(noreturn)
    );
}
