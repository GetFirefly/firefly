///! The memory specific to a process in the VM.
use std::fmt::{self, Debug};
use std::sync::{Mutex, RwLock};

use num_bigint::BigInt;

use crate::binary::{sub, Binary};
use crate::exception::Exception;
use crate::float::Float;
use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::{self, big};
use crate::list::Cons;
use crate::mailbox::Mailbox;
use crate::map::Map;
use crate::message::Message;
use crate::reference;
use crate::term::Term;
use crate::tuple::Tuple;

pub mod identifier;
pub mod local;

pub struct Process {
    pub pid: Term,
    pub registered_name: RwLock<Option<Term>>,
    pub heap: Mutex<Heap>,
    pub mailbox: Mutex<Mailbox>,
}

impl Process {
    #[cfg(test)]
    fn new() -> Self {
        Process {
            pid: identifier::local::next(),
            registered_name: Default::default(),
            heap: Default::default(),
            mailbox: Default::default(),
        }
    }

    pub fn alloc_term_slice(&self, slice: &[Term]) -> *const Term {
        self.heap.lock().unwrap().alloc_term_slice(slice)
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> &'static Cons {
        self.heap.lock().unwrap().cons(head, tail)
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

    pub fn local_reference(&self) -> &'static reference::local::Reference {
        self.heap.lock().unwrap().local_reference()
    }

    #[cfg(test)]
    pub fn number_to_local_reference(&self, number: u64) -> &'static reference::local::Reference {
        self.heap.lock().unwrap().number_to_local_reference(number)
    }

    pub fn num_bigint_big_to_big_integer(&self, big_int: BigInt) -> &'static big::Integer {
        self.heap
            .lock()
            .unwrap()
            .num_bigint_big_to_big_integer(big_int)
    }

    pub fn send_heap_message(&self, heap: Heap, message: Term) {
        self.mailbox
            .lock()
            .unwrap()
            .push(Message::Heap { heap, message });
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

    pub fn u64_to_local_reference(&self, number: u64) -> &'static reference::local::Reference {
        self.heap.lock().unwrap().u64_to_local_reference(number)
    }
}

impl Debug for Process {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.pid)
    }
}

#[cfg(test)]
impl PartialEq for Process {
    fn eq(&self, other: &Process) -> bool {
        self.pid == other.pid
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
