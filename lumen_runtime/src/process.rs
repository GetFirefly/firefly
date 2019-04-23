///! The memory specific to a process in the VM.
use std::sync::Mutex;

use im::hashmap::HashMap;
use num_bigint::BigInt;

use liblumen_arena::TypedArena;

use crate::binary::{heap, sub, Binary};
use crate::exception::Exception;
use crate::float::Float;
use crate::integer::{self, big};
use crate::list::Cons;
use crate::map::Map;
use crate::reference;
use crate::term::Term;
use crate::tuple::Tuple;

pub mod identifier;
pub mod local;

pub struct Process {
    pub pid: Term,
    big_integer_arena: Mutex<TypedArena<big::Integer>>,
    byte_arena: Mutex<TypedArena<u8>>,
    cons_arena: Mutex<TypedArena<Cons>>,
    external_pid_arena: Mutex<TypedArena<identifier::External>>,
    float_arena: Mutex<TypedArena<Float>>,
    heap_binary_arena: Mutex<TypedArena<heap::Binary>>,
    map_arena: Mutex<TypedArena<Map>>,
    local_reference_arena: Mutex<TypedArena<reference::local::Reference>>,
    subbinary_arena: Mutex<TypedArena<sub::Binary>>,
    term_arena: Mutex<TypedArena<Term>>,
}

impl Process {
    #[cfg(test)]
    fn new() -> Self {
        Process {
            pid: identifier::local::next(),
            big_integer_arena: Default::default(),
            byte_arena: Default::default(),
            cons_arena: Default::default(),
            external_pid_arena: Default::default(),
            float_arena: Default::default(),
            heap_binary_arena: Default::default(),
            map_arena: Default::default(),
            local_reference_arena: Default::default(),
            subbinary_arena: Default::default(),
            term_arena: Default::default(),
        }
    }

    pub fn alloc_term_slice(&self, slice: &[Term]) -> *const Term {
        self.term_arena.lock().unwrap().alloc_slice(slice).as_ptr()
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> &'static Cons {
        let pointer = self.cons_arena.lock().unwrap().alloc(Cons::new(head, tail)) as *const Cons;

        unsafe { &*pointer }
    }

    pub fn external_pid(
        &self,
        node: usize,
        number: usize,
        serial: usize,
    ) -> &'static identifier::External {
        let pointer = self
            .external_pid_arena
            .lock()
            .unwrap()
            .alloc(identifier::External::new(node, number, serial))
            as *const identifier::External;

        unsafe { &*pointer }
    }

    pub fn f64_to_float(&self, f: f64) -> &'static Float {
        let pointer = self.float_arena.lock().unwrap().alloc(Float::new(f)) as *const Float;

        unsafe { &*pointer }
    }

    pub fn local_reference(&self) -> &'static reference::local::Reference {
        let pointer = self
            .local_reference_arena
            .lock()
            .unwrap()
            .alloc(reference::local::Reference::next())
            as *const reference::local::Reference;

        unsafe { &*pointer }
    }

    #[cfg(test)]
    pub fn number_to_local_reference(&self, number: u64) -> &'static reference::local::Reference {
        let pointer = self
            .local_reference_arena
            .lock()
            .unwrap()
            .alloc(reference::local::Reference::new(number))
            as *const reference::local::Reference;

        unsafe { &*pointer }
    }

    pub fn num_bigint_big_in_to_big_integer(&self, big_int: BigInt) -> &'static big::Integer {
        let pointer = self
            .big_integer_arena
            .lock()
            .unwrap()
            .alloc(big::Integer::new(big_int)) as *const big::Integer;

        unsafe { &*pointer }
    }

    pub fn subbinary(
        &self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
    ) -> &'static sub::Binary {
        let pointer = self.subbinary_arena.lock().unwrap().alloc(sub::Binary::new(
            original,
            byte_offset,
            bit_offset,
            byte_count,
            bit_count,
        )) as *const sub::Binary;

        unsafe { &*pointer }
    }

    pub fn slice_to_binary(&self, slice: &[u8]) -> Binary {
        // TODO use reference counted binaries for bytes.len() > 64
        let heap_binary = self.slice_to_heap_binary(slice);

        Binary::Heap(heap_binary)
    }

    pub fn slice_to_map(&self, slice: &[(Term, Term)]) -> &Map {
        let mut inner: HashMap<Term, Term> = HashMap::new();

        for (key, value) in slice {
            inner.insert(key.clone(), value.clone());
        }

        let pointer = self.map_arena.lock().unwrap().alloc(Map::new(inner)) as *const Map;

        unsafe { &*pointer }
    }

    pub fn slice_to_tuple(&self, slice: &[Term]) -> &Tuple {
        Tuple::from_slice(slice, &self)
    }

    pub fn u64_to_local_reference(&self, number: u64) -> &'static reference::local::Reference {
        let pointer = self
            .local_reference_arena
            .lock()
            .unwrap()
            .alloc(reference::local::Reference::new(number))
            as *const reference::local::Reference;

        unsafe { &*pointer }
    }

    // Private

    fn slice_to_heap_binary(&self, bytes: &[u8]) -> &'static heap::Binary {
        let locked_byte_arena = self.byte_arena.lock().unwrap();

        let arena_bytes: &[u8] = if bytes.len() != 0 {
            locked_byte_arena.alloc_slice(bytes)
        } else {
            &[]
        };

        let pointer = self
            .heap_binary_arena
            .lock()
            .unwrap()
            .alloc(heap::Binary::new(arena_bytes)) as *const heap::Binary;

        unsafe { &*pointer }
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
