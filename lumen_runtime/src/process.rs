#![cfg_attr(not(test), allow(dead_code))]
///! The memory specific to a process in the VM.
use std::cmp::Ordering;
use std::sync::{Arc, RwLock, Weak};

use num_bigint::BigInt;

use liblumen_arena::TypedArena;

use crate::atom::{self, Existence, Existence::*};
use crate::binary::{heap, sub, Binary};
use crate::environment::Environment;
use crate::exception::{self, Exception};
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

    pub fn atom_index_to_string(&self, atom_index: atom::Index) -> String {
        self.environment
            .upgrade()
            .unwrap()
            .read()
            .unwrap()
            .atom_index_to_string(atom_index)
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

    pub fn str_to_atom_index(
        &mut self,
        name: &str,
        existence: Existence,
    ) -> Result<atom::Index, Exception> {
        self.environment
            .upgrade()
            .unwrap()
            .write()
            .unwrap()
            .str_to_atom_index(name, existence)
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

/// Like `std::fmt::Debug`, but additionally takes `&Process` in case it is needed to lookup
/// values in the process.
pub trait DebugInProcess {
    fn format_in_process(&self, process: &Process) -> String;
}

impl DebugInProcess for exception::Result {
    fn format_in_process(&self, process: &Process) -> String {
        match self {
            Ok(term) => format!("Ok({})", term.format_in_process(process)),
            Err(Exception { class, reason, arguments, file, line, column }) => format!(
                "Err(Exception {{ class: {:?}, reason: {}, arguments: {}, file: {:?}, line: {:?}, column: {:?} }})",
                class, arguments.format_in_process(&process), reason.format_in_process(&process), file, line, column
            ),
        }
    }
}

/// Like `std::cmp::Ord`, but additionally takes `&Process` in case it is needed to lookup
/// values in the process.
pub trait OrderInProcess<Rhs: ?Sized = Self> {
    /// This method returns an ordering between `self` and `other` values.
    #[must_use]
    fn cmp_in_process(&self, other: &Rhs, process: &Process) -> Ordering;
}

impl OrderInProcess for exception::Result {
    fn cmp_in_process(&self, other: &Self, process: &Process) -> Ordering {
        match (self, other) {
            (Ok(self_ok), Ok(other_ok)) => self_ok.cmp_in_process(&other_ok, process),
            (Ok(_), Err(_)) => Ordering::Less,
            (Err(_), Ok(_)) => Ordering::Greater,
            (
                Err(Exception {
                    class: self_class,
                    reason: self_reason,
                    ..
                }),
                Err(Exception {
                    class: other_class,
                    reason: other_reason,
                    ..
                }),
            ) => match self_class.cmp(&other_class) {
                Ordering::Equal => self_reason.cmp_in_process(other_reason, process),
                ordering => ordering,
            },
        }
    }
}

impl OrderInProcess for Vec<Term> {
    fn cmp_in_process(&self, other: &Vec<Term>, process: &Process) -> Ordering {
        assert_eq!(self.len(), other.len());

        let mut final_ordering = Ordering::Equal;

        for (self_element, other_element) in self.iter().zip(other.iter()) {
            match self_element.cmp_in_process(other_element, process) {
                Ordering::Equal => continue,
                ordering => {
                    final_ordering = ordering;

                    break;
                }
            }
        }

        final_ordering
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

#[macro_export]
macro_rules! assert_cmp_in_process {
    ($left:expr, $ordering:expr, $right:expr, $process:expr) => ({
        use std::cmp::Ordering;

        use crate::process::{DebugInProcess, OrderInProcess};

        match (&$left, &$ordering, &$right, &$process) {
            (left_val, ordering_val, right_val, process_val) => {
                if !((*left_val).cmp_in_process(right_val, process_val) == *ordering_val) {
                     let ordering_str = match *ordering_val {
                         Ordering::Less => "<",
                         Ordering::Equal => "==",
                         Ordering::Greater => ">"
                     };
                     panic!(r#"assertion failed: `(left {} right)`
  left: `{}`,
 right: `{}`"#,
                       ordering_str,
                       left_val.format_in_process(process_val),
                       right_val.format_in_process(process_val)
                     )
                }
            }
        }
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr,) => ({
        assert_cmp_in_process!($left, $ordering, $right, $process)
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr, $($arg:tt)+) => ({
        use std::cmp::Ordering;

        use crate::process::{DebugInProcess, OrderInProcess};

        match (&$left, &$ordering, &$right, &$process) {
            (left_val, ordering_val, right_val, process_val) => {
                if !((*left_val).cmp_in_process(right_val, process_val) == *ordering_val) {
                     let ordering_str = match *ordering_val {
                         Ordering::Less => "<",
                         Ordering::Equal => "==",
                         Ordering::Greater => ">"
                     };
                     panic!(r#"assertion failed: `(left {} right)`
  left: `{}`,
 right: `{}`: {}"#,
                       ordering_str,
                       left_val.format_in_process(process_val),
                       right_val.format_in_process(process_val),
                       format_args!($($arg)+)
                     )
                }
            }
        }
    });
}

#[macro_export]
macro_rules! refute_cmp_in_process {
    ($left:expr, $ordering:expr, $right:expr, $process:expr) => ({
        use std::cmp::Ordering;

        use crate::process::{DebugInProcess, OrderInProcess};

        match (&$left, &$ordering, &$right, &$process) {
            (left_val, ordering_val, right_val, process_val) => {
                if (*left_val).cmp_in_process(right_val, process_val) == *ordering_val {
                     let ordering_str = match *ordering_val {
                         Ordering::Less => ">=",
                         Ordering::Equal => "!=",
                         Ordering::Greater => "<="
                     };
                     panic!(r#"assertion failed: `(left {} right)`
  left: `{}`,
 right: `{}`"#,
                       ordering_str,
                       left_val.format_in_process(process_val),
                       right_val.format_in_process(process_val)
                     )
                }
            }
        }
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr,) => ({
        assert_cmp_in_process!($left, $ordering, $right, $process)
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr, $($arg:tt)+) => ({
        use std::cmp::Ordering;

        use crate::process::{DebugInProcess, OrderInProcess};

        match (&$left, &$ordering, &$right, &$process) {
            (left_val, ordering_val, right_val, process_val) => {
                if (*left_val).cmp_in_process(right_val, process_val) == *ordering_val {
                     let ordering_str = match *ordering_val {
                         Ordering::Less => ">=",
                         Ordering::Equal => "!=",
                         Ordering::Greater => "<="
                     };
                     panic!(r#"assertion failed: `(left {} right)`
  left: `{}`,
 right: `{}`: {}"#,
                       ordering_str,
                       left_val.format_in_process(process_val),
                       right_val.format_in_process(process_val),
                       format_args!($($arg)+)
                     )
                }
            }
        }
    });
}

#[macro_export]
macro_rules! assert_eq_in_process {
    ($left:expr, $right:expr, $process:expr) => ({
        assert_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process)
    });
    ($left:expr, $right:expr, $process:expr,) => ({
        assert_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process)
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr, $($arg:tt)+) => ({
        assert_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process, $($arg)+)
    });
}

#[macro_export]
macro_rules! assert_ne_in_process {
    ($left:expr, $right:expr, $process:expr) => ({
        refute_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process)
    });
    ($left:expr, $right:expr, $process:expr,) => ({
        refute_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process)
    });
    ($left:expr, $ordering:expr, $right:expr, $process:expr, $($arg:tt)+) => ({
        refute_cmp_in_process!($left, std::cmp::Ordering::Equal, $right, $process, $($arg)+)
    });
}

/// Like `std::convert::Into`, but additionally takes `&mut Process` in case it is needed to
/// lookup or create new values in the `Process`.
pub trait IntoProcess<T> {
    /// Performs the conversion.
    fn into_process(self, process: &mut Process) -> T;
}

impl IntoProcess<Term> for bool {
    fn into_process(self, mut process: &mut Process) -> Term {
        Term::str_to_atom(&self.to_string(), DoNotCare, &mut process)
            .unwrap()
            .into()
    }
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
            let mut first_process = first_process_rw_lock.write().unwrap();

            let second_process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut second_process = second_process_rw_lock.write().unwrap();

            assert_ne_in_process!(
                erlang::self_0(&first_process),
                erlang::self_0(&second_process),
                first_process
            );
            assert_eq_in_process!(
                erlang::self_0(&first_process),
                Term::local_pid(0, 0, &mut first_process).unwrap(),
                &mut first_process
            );
            assert_eq_in_process!(
                erlang::self_0(&second_process),
                Term::local_pid(1, 0, &mut second_process).unwrap(),
                &mut second_process
            );
        }

        #[test]
        fn number_rolling_over_increments_serial() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();

            let first_process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut first_process = first_process_rw_lock.write().unwrap();

            let mut final_pid = None;

            for _ in 0..identifier::NUMBER_MAX + 1 {
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let process = process_rw_lock.read().unwrap();
                final_pid = Some(erlang::self_0(&process))
            }

            assert_eq_in_process!(
                final_pid.unwrap(),
                Term::local_pid(0, 1, &mut first_process).unwrap(),
                first_process
            );
        }
    }

    mod str_to_atom_index {
        use super::*;

        use crate::environment;

        #[test]
        fn without_same_string_have_different_index() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();

            assert_ne!(
                process.str_to_atom_index("true", DoNotCare).unwrap().0,
                process.str_to_atom_index("false", DoNotCare).unwrap().0
            )
        }

        #[test]
        fn with_same_string_have_same_index() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();

            assert_eq!(
                process.str_to_atom_index("atom", DoNotCare).unwrap().0,
                process.str_to_atom_index("atom", DoNotCare).unwrap().0
            )
        }
    }
}
