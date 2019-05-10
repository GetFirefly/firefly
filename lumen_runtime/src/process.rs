///! The memory specific to a process in the VM.
use std::collections::vec_deque::VecDeque;
use std::fmt::{self, Debug};
use std::sync::{Arc, Mutex, RwLock, Weak};

use num_bigint::BigInt;

use crate::atom::Existence::DoNotCare;
use crate::binary::{sub, Binary};
use crate::exception::Exception;
use crate::float::Float;
use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::{self, big};
use crate::list::Cons;
use crate::mailbox::Mailbox;
use crate::map::Map;
use crate::message::{self, Message};
use crate::process::instruction::Instruction;
use crate::reference;
use crate::scheduler::{self, Priority, Scheduler};
use crate::term::Term;
use crate::tuple::Tuple;

pub mod identifier;
mod instruction;
pub mod local;

pub struct FunctionCalls {
    pub initial: ModuleFunctionArity,
    pub current: ModuleFunctionArity,
}

impl FunctionCalls {
    fn new(module: Term, function: Term, arity: usize) -> FunctionCalls {
        let initial = ModuleFunctionArity {
            module,
            function,
            arity,
        };

        FunctionCalls {
            initial,
            current: initial,
        }
    }
}

// 4000 in [BEAM](https://github.com/erlang/otp/blob/61ebe71042fce734a06382054690d240ab027409/erts/emulator/beam/erl_vm.h#L39)
const MAX_REDUCTIONS: Reductions = 4_000;

#[derive(Clone, Copy)]
pub struct ModuleFunctionArity {
    pub module: Term,
    pub function: Term,
    pub arity: usize,
}

pub struct Process {
    pub scheduler: Mutex<Option<Weak<Scheduler>>>,
    pub priority: Priority,
    #[allow(dead_code)]
    parent_pid: Option<Term>,
    pub pid: Term,
    #[allow(dead_code)]
    function_calls: Option<FunctionCalls>,
    instructions: Mutex<VecDeque<Instruction>>,
    pub registered_name: RwLock<Option<Term>>,
    pub stack: Mutex<Stack>,
    pub status: RwLock<Status>,
    pub heap: Mutex<Heap>,
    pub mailbox: Mutex<Mailbox>,
}

impl Process {
    pub fn init() -> Self {
        Self::new(
            Default::default(),
            None,
            None,
            Default::default(),
            Default::default(),
        )
    }

    pub fn spawn(
        parent_process: &Process,
        module: Term,
        function: Term,
        arguments: Vec<Term>,
    ) -> Self {
        assert!(module.is_atom());
        assert!(function.is_atom());

        let arity = arguments.len();
        let heap: Heap = Default::default();
        let heap_arguments = arguments.clone_into_heap(&heap);

        let mut instructions = VecDeque::new();
        instructions.push_back(Instruction::Apply {
            module,
            function,
            arguments: heap_arguments,
        });

        Self::new(
            parent_process.priority,
            Some(parent_process.pid),
            Some(FunctionCalls::new(module, function, arity)),
            instructions,
            Mutex::new(heap),
        )
    }

    pub fn alloc_term_slice(&self, slice: &[Term]) -> *const Term {
        self.heap.lock().unwrap().alloc_term_slice(slice)
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> &'static Cons {
        self.heap.lock().unwrap().cons(head, tail)
    }

    fn exit(&self) {
        *self.status.write().unwrap() =
            Status::Exiting(exit!(Term::str_to_atom("normal", DoNotCare).unwrap()));
    }

    fn exception(&self, exception: Exception) {
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

    /// Run process until `reductions` exceeds `MAX_REDUCTIONS` or process exits
    pub fn run(arc_process: &Arc<Process>) {
        *arc_process.status.write().unwrap() = Status::Running;

        let mut locked_instructions = arc_process.instructions.lock().unwrap();
        let mut reductions = 0;

        loop {
            match locked_instructions.pop_front() {
                Some(instruction) => {
                    if instruction.run(arc_process) {
                        reductions += 1;

                        if MAX_REDUCTIONS <= reductions {
                            *arc_process.status.write().unwrap() = Status::Runnable;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                None => {
                    arc_process.exit();
                    break;
                }
            }
        }
    }

    pub fn send_heap_message(&self, heap: Heap, message: Term) {
        self.mailbox
            .lock()
            .unwrap()
            .push(Message::Heap(message::Heap { heap, message }));
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

    // Private

    fn new(
        priority: Priority,
        parent_pid: Option<Term>,
        function_calls: Option<FunctionCalls>,
        instructions: VecDeque<Instruction>,
        heap: Mutex<Heap>,
    ) -> Self {
        Process {
            scheduler: Mutex::new(None),
            priority,
            parent_pid,
            pid: identifier::local::next(),
            function_calls,
            instructions: Mutex::new(instructions),
            registered_name: Default::default(),
            status: Default::default(),
            heap,
            stack: Default::default(),
            mailbox: Default::default(),
        }
    }
}

impl Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.pid)?;

        match *self.registered_name.read().unwrap() {
            Some(registered_name) => write!(f, "({:?})", registered_name),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
impl PartialEq for Process {
    fn eq(&self, other: &Process) -> bool {
        self.pid == other.pid
    }
}

type Reductions = u16;

#[derive(Default)]
pub struct Stack(VecDeque<Term>);

impl Stack {
    pub fn get(&self, index: usize) -> Option<&Term> {
        self.0.get(index)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, term: Term) {
        self.0.push_front(term);
    }
}

// [BEAM statuses](https://github.com/erlang/otp/blob/551d03fe8232a66daf1c9a106194aa38ef660ef6/erts/emulator/beam/erl_process.c#L8944-L8972)
#[derive(PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub enum Status {
    Runnable,
    Running,
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
