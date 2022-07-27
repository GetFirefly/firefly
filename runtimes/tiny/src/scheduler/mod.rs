mod queue;

use std::arch::global_asm;
use std::cell::{OnceCell, UnsafeCell};
use std::mem;
use std::ptr;
use std::sync::{atomic::AtomicU64, Arc};
use std::thread::{self, ThreadId};

use liblumen_rt::function::{DynamicCallee, ModuleFunctionArity};
use liblumen_rt::process::{Process, ProcessStatus};
use liblumen_rt::term::{OpaqueTerm, Pid, ProcessId};

use self::queue::RunQueue;

#[thread_local]
pub static CURRENT_PROCESS: UnsafeCell<Option<Arc<Process>>> = UnsafeCell::new(None);

#[thread_local]
pub static CURRENT_SCHEDULER: OnceCell<Scheduler> = OnceCell::new();

/// Returns a reference to the scheduler for the current thread
pub fn with_current<F, R>(fun: F) -> R
where
    F: FnOnce(&Scheduler) -> R,
{
    fun(CURRENT_SCHEDULER.get().unwrap())
}

/// Initializes the scheduler for the current thread, if not already initialized,
/// returning a reference to it
pub fn init<'a>() -> bool {
    CURRENT_SCHEDULER.get_or_init(|| Scheduler::new().unwrap());
    true
}

/// Applies the currently executing process to the given function
pub fn with_current_process<F, R>(fun: F) -> R
where
    F: FnOnce(&Process) -> R,
{
    let p = unsafe { (&*CURRENT_PROCESS.get()).as_deref().unwrap() };
    fun(p)
}

struct SchedulerData {
    process: Arc<Process>,
    registers: UnsafeCell<CalleeSavedRegisters>,
}
impl SchedulerData {
    fn new(process: Arc<Process>) -> Self {
        Self {
            process,
            registers: UnsafeCell::new(Default::default()),
        }
    }

    #[allow(dead_code)]
    fn pid(&self) -> Pid {
        Pid::Local {
            id: self.process.pid(),
        }
    }

    fn registers(&self) -> &CalleeSavedRegisters {
        unsafe { &*self.registers.get() }
    }

    fn registers_mut(&self) -> &mut CalleeSavedRegisters {
        unsafe { &mut *self.registers.get() }
    }
}
unsafe impl Send for SchedulerData {}
unsafe impl Sync for SchedulerData {}

pub struct Scheduler {
    pub id: ThreadId,
    // References are always 64-bits even on 32-bit platforms
    #[allow(dead_code)]
    next_reference_id: AtomicU64,
    // In this runtime, we aren't doing work-stealing, so the run queue
    // is never accessed by any other thread
    run_queue: UnsafeCell<RunQueue>,
    prev: UnsafeCell<Option<Arc<SchedulerData>>>,
    current: UnsafeCell<Arc<SchedulerData>>,
}
// This guarantee holds as long as `init` and `current` are only
// ever accessed by the scheduler when scheduling
unsafe impl Sync for Scheduler {}
impl Scheduler {
    /// Creates a new scheduler with the default configuration
    fn new() -> anyhow::Result<Self> {
        let id = thread::current().id();

        // The root process is how the scheduler gets time for itself,
        // and is also how we know when to shutdown the scheduler due
        // to termination of all its processes
        let root = {
            let process = Arc::new(Process::new(
                None,
                ProcessId::next(),
                "root:init/0".parse().unwrap(),
            ));
            unsafe {
                process.set_status(ProcessStatus::Running);
            }
            let mut registers = CalleeSavedRegisters::default();
            unsafe {
                registers.set(1, 0x0u64);
            }
            Arc::new(SchedulerData {
                process,
                registers: UnsafeCell::new(registers),
            })
        };

        // The scheduler starts with the root process running
        Ok(Self {
            id,
            next_reference_id: AtomicU64::new(0),
            run_queue: UnsafeCell::new(RunQueue::default()),
            prev: UnsafeCell::new(None),
            current: UnsafeCell::new(root),
        })
    }

    fn parent(&self) -> ProcessId {
        self.current().process.pid()
    }

    fn prev(&self) -> &SchedulerData {
        unsafe { (&*self.prev.get()).as_deref().unwrap() }
    }

    fn take_prev(&self) -> Arc<SchedulerData> {
        unsafe { (&mut *self.prev.get()).take().unwrap() }
    }

    fn current(&self) -> &SchedulerData {
        unsafe { &*self.current.get() }
    }

    pub fn current_process(&self) -> Arc<Process> {
        self.current().process.clone()
    }

    /// Swaps the prev and current scheduler data in-place and updates CURRENT_PROCESS
    ///
    /// This is intended for use when yielding to the scheduler
    fn swap_current(&self) {
        let prev = unsafe { (&mut *self.prev.get()).as_mut().unwrap() };
        // Change the previous process status to Runnable
        unsafe {
            let prev_status = prev.process.status();
            if prev_status == ProcessStatus::Running {
                prev.process.set_status(ProcessStatus::Runnable);
            }
        }
        let proc = prev.process.clone();
        let current = unsafe { &mut *self.current.get() };
        mem::swap(prev, current);
        let _ = unsafe { (&mut *CURRENT_PROCESS.get()).replace(proc) };
    }

    /// Swaps the current scheduler data with the one provided, and updates CURRENT_PROCESS
    ///
    /// This is intended for use when swapping from the scheduler to a process, and as such,
    /// it requires that `prev` contains None when this is called
    fn swap_with(&self, new: Arc<SchedulerData>) {
        assert!(unsafe { (&mut *self.prev.get()).replace(new).is_none() });
        self.swap_current()
    }

    /// Returns true if the root process (scheduler) is running
    #[allow(dead_code)]
    fn is_root(&self) -> bool {
        unsafe { (&*self.prev.get()).is_none() }
    }

    pub(super) fn spawn_init(&self) -> anyhow::Result<Arc<Process>> {
        // The init process is the actual "root" Erlang process, it acts
        // as the entry point for the program from Erlang's perspective,
        // and is responsible for starting/stopping the system in Erlang.
        //
        // If this process exits, the scheduler terminates
        let mfa: ModuleFunctionArity = "init:start/0".parse().unwrap();
        //let init_fn = function::find_symbol(&mfa).expect("unable to locate init:start/0 function!");
        let init_fn = crate::init::start as DynamicCallee;
        let process = Arc::new(Process::new(Some(self.parent()), ProcessId::next(), mfa));

        let data = Arc::new(SchedulerData::new(process));

        Self::runnable(&data, init_fn);

        Ok(self.schedule(data))
    }

    fn schedule(&self, data: Arc<SchedulerData>) -> Arc<Process> {
        let handle = data.process.clone();
        let rq = unsafe { &mut *self.run_queue.get() };
        rq.schedule(data);
        handle
    }

    #[inline]
    pub(super) fn run_once(&self) -> bool {
        // The scheduler will yield to a process to execute
        self.scheduler_yield()
    }

    fn runnable(scheduler: &SchedulerData, init_fn: DynamicCallee) {
        #[derive(Copy, Clone)]
        struct StackPointer(*mut u64);
        impl StackPointer {
            #[inline(always)]
            unsafe fn push(&mut self, value: u64) {
                self.0 = self.0.offset(-1);
                ptr::write(self.0, value);
            }
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
            scheduler.process.set_status(ProcessStatus::Runnable);

            let stack = scheduler.process.stack();
            let registers = scheduler.registers_mut();
            // This can be used to push items on the process
            // stack before it starts executing. For now that
            // is not being done
            let mut sp = StackPointer(stack.top as *mut u64);
            // Make room for the CFA above the frame pointer, and add padding so
            // stack alignment of 16 bytes is preserved
            sp.push(0);
            sp.push(0);

            // Write stack/frame pointer initial values
            registers.set_stack_pointer(sp.0 as u64);
            registers.set_frame_pointer(sp.0 as u64);

            // TODO: Set up for closures
            // If the init function is a closure, place it in
            // the first callee-save register, which will be moved to
            // the first argument register (e.g. %rsi) by swap_stack for
            // the call to the entry point
            registers.set(0, OpaqueTerm::NONE);

            // This is used to indicate to swap_stack that this process
            // is being swapped to for the first time, which allows the
            // function to perform some initial one-time setup to link
            // call frames for the unwinder and call the entry point
            registers.set(1, FIRST_SWAP);

            // The function that swap_stack will call as entry
            registers.set(2, init_fn as u64);
        }
    }

    // TODO: Request application master termination for controlled shutdown
    // This request will always come from the thread which spawned the application
    // master, i.e. the "main" scheduler thread
    //
    // Returns `Ok(())` if shutdown was successful, `Err(anyhow::Error)` if something
    // went wrong during shutdown, and it was not able to complete normally
    pub(super) fn shutdown(&self) -> anyhow::Result<()> {
        // For now just Ok(()), but this needs to be addressed when proper
        // system startup/shutdown is in place
        Ok(())
    }

    pub(super) fn process_yield(&self) -> bool {
        // Swap back to the scheduler, which is currently "suspended" in `prev`.
        // When `swap_stack` is called it will look like a return from the last call
        // to `swap_stack` from the scheduler loop.
        //
        // This function will appear to return normally to the caller if the process
        // that yielded is rescheduled
        let prev = unsafe { (&*(self.prev.get())).as_deref().unwrap().registers() };
        let current = unsafe { (&*(self.current.get())).registers_mut() };

        unsafe {
            swap_stack(current, prev, FIRST_SWAP);
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
        loop {
            let next = {
                let rq = unsafe { &mut *self.run_queue.get() };
                rq.next()
            };

            match next {
                Some(scheduler_data) => {
                    // Found a process to schedule
                    unsafe {
                        // The swap takes care of setting up the to-be-scheduled process
                        // as the current process, and swaps to its stack. The code below
                        // is executed when that process has yielded and we're resetting
                        // the state of the scheduler such that the "current process" is
                        // the scheduler itself
                        self.swap_process(scheduler_data);
                    }
                    // When we reach here, the process has yielded
                    // back to the scheduler, and is still marked
                    // as the current process. We need to handle
                    // swapping it out with the scheduler process
                    // and handling its exit, if exiting
                    self.swap_current();
                    let prev = self.take_prev();
                    match prev.process.status() {
                        ProcessStatus::Running => {
                            let rq = unsafe { &mut *self.run_queue.get() };
                            rq.reschedule(prev);
                        }
                        ProcessStatus::Exiting => {
                            // TODO
                        }
                        _ => (),
                    }

                    // When reached, either the process scheduled is the root process,
                    // or the process is exiting and we called .reduce(); either way we're
                    // returning to the main scheduler loop to check for signals, etc.
                    break true;
                }
                None => {
                    // Nothing to schedule, so bail
                    break false;
                }
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
    unsafe fn swap_process(&self, new: Arc<SchedulerData>) {
        // Mark the new process as Running
        new.process.set_status(ProcessStatus::Running);

        self.swap_with(new);
        let prev = self.prev();
        let new = self.current();

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
        swap_stack(prev.registers_mut(), new.registers(), FIRST_SWAP);
    }
}

#[derive(Default, Debug)]
#[repr(C)]
#[cfg(all(unix, target_arch = "x86_64"))]
struct CalleeSavedRegisters {
    pub rsp: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbx: u64,
    pub rbp: u64,
}
#[cfg(target_arch = "x86_64")]
impl CalleeSavedRegisters {
    #[inline(always)]
    unsafe fn set<T: Copy>(&mut self, index: isize, value: T) {
        let base = std::ptr::addr_of!(self.rbp);
        let base = base.offset((-index) - 2) as *mut T;
        base.write(value);
    }

    #[inline(always)]
    unsafe fn set_stack_pointer(&mut self, value: u64) {
        self.rsp = value;
    }

    #[inline(always)]
    unsafe fn set_frame_pointer(&mut self, value: u64) {
        self.rbp = value;
    }
}

#[derive(Debug, Default)]
#[repr(C)]
#[cfg(all(unix, target_arch = "aarch64"))]
struct CalleeSavedRegisters {
    pub sp: u64,
    pub x29: u64,
    pub x28: u64,
    pub x27: u64,
    pub x26: u64,
    pub x25: u64,
    pub x24: u64,
    pub x23: u64,
    pub x22: u64,
    pub x21: u64,
    pub x20: u64,
    pub x19: u64,
}
#[cfg(target_arch = "aarch64")]
impl CalleeSavedRegisters {
    #[inline(always)]
    unsafe fn set<T: Copy>(&mut self, index: isize, value: T) {
        let base = std::ptr::addr_of!(self.x19);
        let base = base.offset(-index) as *mut T;
        base.write(value);
    }

    #[inline(always)]
    unsafe fn set_stack_pointer(&mut self, value: u64) {
        self.sp = value;
    }

    #[inline(always)]
    unsafe fn set_frame_pointer(&mut self, value: u64) {
        self.x29 = value;
    }
}

const FIRST_SWAP: u64 = 0xdeadbeef;

extern "C-unwind" {
    #[link_name = "__lumen_swap_stack"]
    fn swap_stack(
        prev: *mut CalleeSavedRegisters,
        new: *const CalleeSavedRegisters,
        first_swap: u64,
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
global_asm!(include_str!("swap_stack/swap_stack_linux_x86_64.s"));
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
global_asm!(include_str!("swap_stack/swap_stack_macos_x86_64.s"));
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
global_asm!(include_str!("swap_stack/swap_stack_macos_aarch64.s"));
