///! The memory specific to a process in the VM.
use std::collections::HashMap;
#[cfg(debug_assertions)]
use std::fmt::Debug;
use std::fmt::{self, Display};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard, Weak};

use num_bigint::BigInt;

use crate::atom::Existence::DoNotCare;
use crate::binary::{sub, Binary};
use crate::code::{self, Code};
use crate::exception::{self, Exception};
use crate::float::Float;
use crate::function::Function;
use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::{self, big};
use crate::list::Cons;
use crate::mailbox::Mailbox;
use crate::map::Map;
use crate::message::{self, Message};
use crate::process::stack::frame::Frame;
use crate::process::stack::Stack;
use crate::reference;
use crate::registry::*;
use crate::scheduler::{self, Priority, Scheduler};
use crate::term::Term;
use crate::tuple::Tuple;
use std::hash::{Hash, Hasher};

pub mod identifier;
pub mod local;
pub mod stack;

// 4000 in [BEAM](https://github.com/erlang/otp/blob/61ebe71042fce734a06382054690d240ab027409/erts/emulator/beam/erl_vm.h#L39)
pub const MAX_REDUCTIONS_PER_RUN: Reductions = 4_000;

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct ModuleFunctionArity {
    pub module: Term,
    pub function: Term,
    pub arity: usize,
}

impl Display for ModuleFunctionArity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            ":{}.{}/{}",
            unsafe { self.module.atom_to_string() },
            unsafe { self.function.atom_to_string() },
            self.arity
        )
    }
}

pub struct Process {
    pub scheduler: Mutex<Option<Weak<Scheduler>>>,
    pub priority: Priority,
    #[allow(dead_code)]
    parent_pid: Option<Term>,
    pub pid: Term,
    #[allow(dead_code)]
    initial_module_function_arity: Arc<ModuleFunctionArity>,
    /// The number of reductions in the current `run`.  `code` MUST return when `run_reductions`
    /// exceeds `MAX_REDUCTIONS_PER_RUN`.
    run_reductions: AtomicU16,
    pub total_reductions: AtomicU64,
    pub registered_name: RwLock<Option<Term>>,
    stack: Mutex<Stack>,
    pub status: RwLock<Status>,
    pub heap: Mutex<Heap>,
    pub mailbox: Mutex<Mailbox>,
}

impl Process {
    pub fn init() -> Self {
        let init = Term::str_to_atom("init", DoNotCare).unwrap();
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module: init,
            function: init,
            arity: 0,
        });
        let frame = Frame::new(Arc::clone(&module_function_arity), code::init);

        let mut stack: Stack = Default::default();
        stack.push(frame);

        Self::new(
            Default::default(),
            None,
            module_function_arity,
            Default::default(),
            stack,
        )
    }

    pub fn spawn(
        parent_process: &Process,
        module: Term,
        function: Term,
        arguments: Term,
        code: Code,
    ) -> Self {
        assert!(module.is_atom());
        assert!(function.is_atom());

        let arity = match arguments.count() {
            Some(count) => count,
            None => {
                #[cfg(debug_assertions)]
                panic!(
                    "Arguments {:?} are neither an empty nor a proper list",
                    arguments
                );
                #[cfg(not(debug_assertions))]
                panic!("Arguments are neither an empty nor a proper list");
            }
        };

        let heap = Default::default();
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let mut frame = Frame::new(Arc::clone(&module_function_arity), code);

        let heap_arguments = arguments.clone_into_heap(&heap);
        frame.push(heap_arguments);

        // Don't need to be cloned into heap because they are atoms
        frame.push(function);
        frame.push(module);

        let mut stack: Stack = Default::default();
        stack.push(frame);

        Self::new(
            parent_process.priority,
            Some(parent_process.pid),
            module_function_arity,
            heap,
            stack,
        )
    }

    pub fn alloc_term_slice(&self, slice: &[Term]) -> *const Term {
        self.heap.lock().unwrap().alloc_term_slice(slice)
    }

    /// Adds a frame for the BIF so that stacktraces include the BIF
    pub fn tail_call_bif<F>(
        arc_process: &Arc<Process>,
        module: Term,
        function: Term,
        arity: usize,
        bif: F,
    ) where
        F: Fn() -> exception::Result,
    {
        let module_function_arity = Arc::new(ModuleFunctionArity {
            module,
            function,
            arity,
        });
        let frame = Frame::new(
            module_function_arity,
            // fake code
            code::apply_fn(),
        );

        // fake frame show BIF shows in stacktraces
        arc_process.push_frame(frame);

        match bif() {
            Ok(term) => {
                arc_process.reduce();

                // remove BIF frame before returning from call, so that caller's caller is invoked
                // by `call_code`
                {
                    let mut locked_stack = arc_process.stack.lock().unwrap();
                    locked_stack.pop().unwrap();
                }
                arc_process.return_from_call(term);

                Self::call_code(arc_process)
            }
            Err(exception) => {
                arc_process.reduce();
                arc_process.exception(exception)
            }
        }
    }

    /// Calls top `Frame`'s `Code` if it exists and the process is not reduced.
    pub fn call_code(arc_process: &Arc<Process>) {
        if !arc_process.is_reduced() {
            let option_code = arc_process
                .stack
                .lock()
                .unwrap()
                .get(0)
                .map(|frame| frame.code());

            match option_code {
                Some(code) => code(arc_process),
                None => (),
            }
        }
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> &'static Cons {
        self.heap.lock().unwrap().cons(head, tail)
    }

    pub fn current_module_function_arity(&self) -> Option<Arc<ModuleFunctionArity>> {
        self.stack
            .lock()
            .unwrap()
            .get(0)
            .map(|frame| frame.module_function_arity())
    }

    pub fn exit(&self) {
        self.reduce();
        self.exception(exit!(Term::str_to_atom("normal", DoNotCare).unwrap()));
    }

    pub fn exception(&self, exception: Exception) {
        *self.status.write().unwrap() = Status::Exiting(exception);
    }

    pub fn external_pid(
        &self,
        node: usize,
        number: usize,
        serial: usize,
    ) -> &'static identifier::External {
        self.heap.lock().unwrap().external_pid(node, number, serial)
    }

    pub fn f64_to_float(&self, f: f64) -> &'static Float {
        self.heap.lock().unwrap().f64_to_float(f)
    }

    pub fn function(
        &self,
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
    ) -> &'static Function {
        self.heap
            .lock()
            .unwrap()
            .function(module_function_arity, code)
    }

    pub fn local_reference(
        &self,
        scheduler_id: &scheduler::ID,
        number: reference::local::Number,
    ) -> &'static reference::local::Reference {
        self.heap
            .lock()
            .unwrap()
            .local_reference(scheduler_id, number)
    }

    pub fn num_bigint_big_to_big_integer(&self, big_int: BigInt) -> &'static big::Integer {
        self.heap
            .lock()
            .unwrap()
            .num_bigint_big_to_big_integer(big_int)
    }

    /// Pops `n` data from top `Frame` of `Stack`.
    ///
    /// Panics if stack does not have enough entries or there are any remain items on the stack.
    pub fn pop_arguments(&self, count: usize) -> Vec<Term> {
        let mut argument_vec: Vec<Term> = Vec::with_capacity(count);
        let mut locked_stack = self.stack.lock().unwrap();

        let frame = locked_stack.get_mut(0).unwrap();

        for _ in 0..count {
            argument_vec.push(frame.pop().unwrap());
        }
        assert!(frame.is_empty());

        argument_vec
    }

    pub fn push_frame(&self, frame: Frame) {
        self.stack.lock().unwrap().push(frame)
    }

    pub fn reduce(&self) {
        self.run_reductions.fetch_add(1, Ordering::SeqCst);
    }

    pub fn is_reduced(&self) -> bool {
        MAX_REDUCTIONS_PER_RUN <= self.run_reductions.load(Ordering::SeqCst)
    }

    pub fn register(ref_arc_process: &Arc<Process>, name: Term) -> Result<Term, Exception> {
        Arc::clone(ref_arc_process).register_in(RW_LOCK_REGISTERED_BY_NAME.write().unwrap(), name)
    }

    pub fn register_in(
        self: Arc<Process>,
        mut writable_registry: RwLockWriteGuard<HashMap<Term, Registered>>,
        name: Term,
    ) -> Result<Term, Exception> {
        let mut writable_registered_name = self.registered_name.write().unwrap();

        match *writable_registered_name {
            None => {
                writable_registry.insert(name, Registered::Process(Arc::downgrade(&self)));
                *writable_registered_name = Some(name);

                Ok(true.into())
            }
            Some(_) => Err(badarg!()),
        }
    }

    pub fn replace_frame(&self, frame: Frame) {
        let mut locked_stack = self.stack.lock().unwrap();

        // unwrap to ensure there is a frame to replace
        locked_stack.pop().unwrap();

        locked_stack.push(frame);
    }

    pub fn return_from_call(&self, term: Term) {
        let mut locked_stack = self.stack.lock().unwrap();

        // remove current frame.  The caller becomes the top frame, so it's `module_function_arity`
        // will be returned from `current_module_function_arity`.
        locked_stack.pop();

        match locked_stack.get_mut(0) {
            Some(caller_frame) => {
                caller_frame.push(term);
            }
            // no caller, return value is thrown away, process will exit when `Scheduler.run_once`
            // detects it has no frames.
            None => (),
        }
    }

    /// Run process until `reductions` exceeds `MAX_REDUCTIONS` or process exits
    pub fn run(arc_process: &Arc<Process>) {
        arc_process.start_running();

        // `code` is expected to set `code` before it returns to be the next spot to continue
        let option_code = arc_process
            .stack
            .lock()
            .unwrap()
            .get(0)
            .map(|frame| frame.code());

        match option_code {
            Some(code) => code(arc_process),
            None => arc_process.exit(),
        }

        arc_process.stop_running();
    }

    #[cfg(test)]
    pub fn stack_len(&self) -> usize {
        self.stack.lock().unwrap().len()
    }

    pub fn stacktrace(&self) -> Vec<Arc<ModuleFunctionArity>> {
        let locked_stack = self.stack.lock().unwrap();
        let mut stacktrace = Vec::with_capacity(locked_stack.len());

        for frame in locked_stack.iter() {
            stacktrace.push(frame.module_function_arity())
        }

        stacktrace
    }

    #[cfg(test)]
    pub fn print_stack(&self) {
        println!("{:?}", self.stack.lock().unwrap());
    }

    fn start_running(&self) {
        *self.status.write().unwrap() = Status::Running;
    }

    fn stop_running(&self) {
        self.total_reductions.fetch_add(
            self.run_reductions.load(Ordering::SeqCst) as u64,
            Ordering::SeqCst,
        );
        self.run_reductions.store(0, Ordering::SeqCst);

        let mut writable_status = self.status.write().unwrap();

        if *writable_status == Status::Running {
            *writable_status = Status::Runnable
        }
    }

    pub fn send_heap_message(&self, heap: Heap, message: Term) {
        self.mailbox
            .lock()
            .unwrap()
            .push(Message::Heap(message::Heap {
                heap,
                term: message,
            }));
    }

    pub fn send_from_self(&self, message: Term) {
        self.mailbox.lock().unwrap().push(Message::Process(message));
    }

    pub fn send_from_other(&self, message: Term) {
        match self.heap.try_lock() {
            Ok(ref mut destination_heap) => {
                let destination_message = message.clone_into_heap(destination_heap);

                self.mailbox
                    .lock()
                    .unwrap()
                    .push(Message::Process(destination_message));
            }
            Err(_) => {
                let heap: Heap = Default::default();
                let heap_message = message.clone_into_heap(&heap);

                self.send_heap_message(heap, heap_message);
            }
        }

        // status.write() scope
        let stop_waiting = {
            let mut writable_status = self.status.write().unwrap();

            if *writable_status == Status::Waiting {
                *writable_status = Status::Runnable;

                true
            } else {
                false
            }
        };

        if stop_waiting {
            let locked_option_weak_scheduler = self.scheduler.lock().unwrap();

            match *locked_option_weak_scheduler {
                Some(ref weak_scheduler) => match weak_scheduler.upgrade() {
                    Some(arc_scheduler) => arc_scheduler.stop_waiting(self),
                    None => unreachable!("Scheduler was Dropped"),
                },
                None => unreachable!("Process not scheduled"),
            }
        }
    }

    pub fn subbinary(
        &self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
    ) -> &'static sub::Binary {
        self.heap.lock().unwrap().subbinary(
            original,
            byte_offset,
            bit_offset,
            byte_count,
            bit_count,
        )
    }

    pub fn slice_to_binary(&self, slice: &[u8]) -> Binary<'static> {
        self.heap.lock().unwrap().slice_to_binary(slice)
    }

    pub fn slice_to_map(&self, slice: &[(Term, Term)]) -> &'static Map {
        self.heap.lock().unwrap().slice_to_map(slice)
    }

    pub fn slice_to_tuple(&self, slice: &[Term]) -> &'static Tuple {
        self.heap.lock().unwrap().slice_to_tuple(slice)
    }

    /// Puts the process in the waiting status
    pub fn wait(self: &Process) {
        *self.status.write().unwrap() = Status::Waiting;
        self.run_reductions.fetch_add(1, Ordering::SeqCst);
    }

    // Private

    fn new(
        priority: Priority,
        parent_pid: Option<Term>,
        initial_module_function_arity: Arc<ModuleFunctionArity>,
        heap: Heap,
        stack: Stack,
    ) -> Self {
        Process {
            scheduler: Mutex::new(None),
            priority,
            parent_pid,
            pid: identifier::local::next(),
            initial_module_function_arity,
            run_reductions: Default::default(),
            total_reductions: Default::default(),
            registered_name: Default::default(),
            status: Default::default(),
            heap: Mutex::new(heap),
            stack: Mutex::new(stack),
            mailbox: Default::default(),
        }
    }
}

#[cfg(debug_assertions)]
impl Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.pid)?;

        match *self.registered_name.read().unwrap() {
            Some(registered_name) => write!(f, " ({:?})", registered_name),
            None => Ok(()),
        }
    }
}

impl Display for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO external pids
        let (number, serial) = unsafe { self.pid.decompose_local_pid() };
        write!(f, "#PID<0.{}.{}>", number, serial)?;

        match *self.registered_name.read().unwrap() {
            Some(registered_name) => write!(f, "({})", unsafe { registered_name.atom_to_string() }),
            None => Ok(()),
        }
    }
}

impl Eq for Process {}

impl Hash for Process {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pid.hash(state);
    }
}

impl PartialEq for Process {
    fn eq(&self, other: &Process) -> bool {
        self.pid == other.pid
    }
}

type Reductions = u16;

// [BEAM statuses](https://github.com/erlang/otp/blob/551d03fe8232a66daf1c9a106194aa38ef660ef6/erts/emulator/beam/erl_process.c#L8944-L8972)
#[derive(PartialEq)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub enum Status {
    Runnable,
    Running,
    Waiting,
    Exiting(Exception),
}

impl Default for Status {
    fn default() -> Status {
        Status::Runnable
    }
}

pub trait TryFromInProcess<T>: Sized {
    fn try_from_in_process(value: T, process: &Process) -> Result<Self, Exception>;
}

pub trait TryIntoInProcess<T>: Sized {
    fn try_into_in_process(self, process: &Process) -> Result<T, Exception>;
}

impl<T, U> TryIntoInProcess<U> for T
where
    U: TryFromInProcess<T>,
{
    fn try_into_in_process(self, process: &Process) -> Result<U, Exception> {
        U::try_from_in_process(self, process)
    }
}

/// Like `std::convert::Into`, but additionally takes `&Process` in case it is needed to
/// lookup or create new values in the `Process`.
pub trait IntoProcess<T> {
    /// Performs the conversion.
    fn into_process(self, process: &Process) -> T;
}

impl IntoProcess<Term> for BigInt {
    fn into_process(self, process: &Process) -> Term {
        let integer: integer::Integer = self.into();

        integer.into_process(process)
    }
}
