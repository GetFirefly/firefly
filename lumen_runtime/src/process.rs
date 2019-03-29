#![cfg_attr(not(test), allow(dead_code))]
///! The memory specific to a process in the VM.
use std::sync::{Arc, RwLock, Weak};

use num_bigint::BigInt;

use liblumen_arena::TypedArena;

use crate::binary::{heap, sub, Binary};
use crate::environment::Environment;
use crate::exception::Exception;
use crate::float::Float;
use crate::integer::{self, big};
use crate::list::Cons;
use crate::map::Map;
use crate::reference::local;
use crate::term::Term;
use crate::tuple::Tuple;

pub mod identifier;

pub struct Process {
    // parent pointer, so must be held weakly to prevent cycle with this field and
    // `Environment.process_by_pid`.
    #[allow(dead_code)]
    environment: Weak<RwLock<Environment>>,
    pub pid: Term,
    big_integer_arena: TypedArena<big::Integer>,
    pub byte_arena: TypedArena<u8>,
    cons_arena: TypedArena<Cons>,
    external_pid_arena: TypedArena<identifier::External>,
    float_arena: TypedArena<Float>,
    pub heap_binary_arena: TypedArena<heap::Binary>,
    pub map_arena: TypedArena<Map>,
    local_reference_arena: TypedArena<local::Reference>,
    pub subbinary_arena: TypedArena<sub::Binary>,
    pub term_arena: TypedArena<Term>,
}

impl Process {
    pub fn new(environment: Arc<RwLock<Environment>>) -> Self {
        Process {
            environment: Arc::downgrade(&Arc::clone(&environment)),
            pid: environment.write().unwrap().next_pid(),
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

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&mut self, head: Term, tail: Term) -> &'static Cons {
        let pointer = self.cons_arena.alloc(Cons::new(head, tail)) as *const Cons;

        unsafe { &*pointer }
    }

    pub fn external_pid(
        &mut self,
        node: usize,
        number: usize,
        serial: usize,
    ) -> &'static identifier::External {
        let pointer = self
            .external_pid_arena
            .alloc(identifier::External::new(node, number, serial))
            as *const identifier::External;

        unsafe { &*pointer }
    }

    pub fn f64_to_float(&self, f: f64) -> &'static Float {
        let pointer = self.float_arena.alloc(Float::new(f)) as *const Float;

        unsafe { &*pointer }
    }

    pub fn local_reference(&mut self) -> &'static local::Reference {
        let pointer =
            self.local_reference_arena.alloc(local::Reference::next()) as *const local::Reference;

        unsafe { &*pointer }
    }

    pub fn num_bigint_big_in_to_big_integer(&self, big_int: BigInt) -> &'static big::Integer {
        let pointer =
            self.big_integer_arena.alloc(big::Integer::new(big_int)) as *const big::Integer;

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
        let pointer = self.subbinary_arena.alloc(sub::Binary::new(
            original,
            byte_offset,
            bit_offset,
            byte_count,
            bit_count,
        )) as *const sub::Binary;

        unsafe { &*pointer }
    }

    pub fn slice_to_binary(&mut self, slice: &[u8]) -> Binary {
        Binary::from_slice(slice, self)
    }

    pub fn slice_to_map(&mut self, slice: &[(Term, Term)]) -> &Map {
        Map::from_slice(slice, self)
    }

    pub fn slice_to_tuple(&mut self, slice: &[Term]) -> &Tuple {
        Tuple::from_slice(slice, &mut self.term_arena)
    }

    pub fn u64_to_local_reference(&mut self, number: u64) -> &'static local::Reference {
        let pointer = self
            .local_reference_arena
            .alloc(local::Reference::new(number)) as *const local::Reference;

        unsafe { &*pointer }
    }
}

pub trait TryFromInProcess<T>: Sized {
    fn try_from_in_process(value: T, process: &mut Process) -> Result<Self, Exception>;
}

pub trait TryIntoInProcess<T>: Sized {
    fn try_into_in_process(self, process: &mut Process) -> Result<T, Exception>;
}

impl<T, U> TryIntoInProcess<U> for T
where
    U: TryFromInProcess<T>,
{
    fn try_into_in_process(self, process: &mut Process) -> Result<U, Exception> {
        U::try_from_in_process(self, process)
    }
}

/// Like `std::convert::Into`, but additionally takes `&mut Process` in case it is needed to
/// lookup or create new values in the `Process`.
pub trait IntoProcess<T> {
    /// Performs the conversion.
    fn into_process(self, process: &mut Process) -> T;
}

impl IntoProcess<Term> for BigInt {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: integer::Integer = self.into();

        integer.into_process(&mut process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod pid {
        use super::*;

        use crate::environment;
        use crate::otp::erlang;

        #[test]
        fn different_processes_in_same_environment_have_different_pids() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();

            let first_process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let first_process = first_process_rw_lock.write().unwrap();

            let second_process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let second_process = second_process_rw_lock.write().unwrap();

            assert_ne!(
                erlang::self_0(&first_process),
                erlang::self_0(&second_process)
            );
            assert_eq!(
                erlang::self_0(&first_process),
                Term::local_pid(0, 0).unwrap()
            );
            assert_eq!(
                erlang::self_0(&second_process),
                Term::local_pid(1, 0).unwrap()
            );
        }

        #[test]
        fn number_rolling_over_increments_serial() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();

            let _ = environment::process(Arc::clone(&environment_rw_lock));

            let mut final_pid = None;

            for _ in 0..identifier::NUMBER_MAX + 1 {
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let process = process_rw_lock.read().unwrap();
                final_pid = Some(erlang::self_0(&process))
            }

            assert_eq!(final_pid.unwrap(), Term::local_pid(0, 1).unwrap());
        }
    }
}
