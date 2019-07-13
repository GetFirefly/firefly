use core::alloc::AllocErr;
use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::default::Default;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem;

use liblumen_core::locks::RwLock;

use lazy_static::lazy_static;

use crate::borrow::CloneToProcess;
use crate::erts::{HeapAlloc, Node};

use super::{AsTerm, Term};
use crate::erts::term::{TypeError, TypedTerm};

/// Generates the next `Pid`.  `Pid`s are not reused for the lifetime of the VM.
pub fn next() -> Pid {
    let mut writable_counter = RW_LOCK_COUNTER.write();

    writable_counter.next()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Pid(usize);
impl Pid {
    /// Given a the raw pid value (as a usize), reifies it into a `Pid`
    #[inline]
    pub unsafe fn from_raw(pid: usize) -> Self {
        Self(pid)
    }

    pub fn new(number: usize, serial: usize) -> Result<Pid, OutOfRange> {
        Self::check(number, serial)
            .map(|(number, serial)| unsafe { Self::new_unchecked(number, serial) })
    }

    pub fn check(number: usize, serial: usize) -> Result<(usize, usize), OutOfRange> {
        if number <= Self::NUMBER_MAX {
            if serial <= Self::SERIAL_MAX {
                Ok((number, serial))
            } else {
                Err(OutOfRange::Serial)
            }
        } else {
            Err(OutOfRange::Number)
        }
    }

    unsafe fn new_unchecked(number: usize, serial: usize) -> Pid {
        Self::from_raw((serial << (Self::NUMBER_BIT_COUNT as usize)) | number)
    }

    pub fn number(&self) -> usize {
        self.0 & Self::NUMBER_MASK
    }

    pub fn serial(&self) -> usize {
        (self.0 & Self::SERIAL_MASK) >> Self::NUMBER_BIT_COUNT
    }

    const NUMBER_BIT_COUNT: u8 = 15;
    const NUMBER_MASK: usize = 0x111_1111_1111_1111;
    pub const NUMBER_MAX: usize = (1 << (Self::NUMBER_BIT_COUNT as usize)) - 1;

    const SERIAL_BIT_COUNT: u8 =
        (mem::size_of::<usize>() * 8 - (Self::NUMBER_BIT_COUNT as usize) - 2) as u8;
    const SERIAL_MASK: usize = !Self::NUMBER_MASK;
    pub const SERIAL_MAX: usize = (1 << (Self::SERIAL_BIT_COUNT as usize)) - 1;
}

unsafe impl AsTerm for Pid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_pid(self.0)
    }
}
impl PartialEq<ExternalPid> for Pid {
    #[inline]
    fn eq(&self, _other: &ExternalPid) -> bool {
        false
    }
}
impl PartialOrd<ExternalPid> for Pid {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.pid)
    }
}

pub struct ExternalPid {
    header: Term,
    node: Node,
    next: *mut u8, // off heap header
    pid: Pid,
}
unsafe impl AsTerm for ExternalPid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for ExternalPid {
    fn clone_to_heap<A: HeapAlloc>(&self, _heap: &mut A) -> Result<Term, AllocErr> {
        unimplemented!()
    }
}

impl Hash for ExternalPid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.pid.hash(state);
    }
}

impl PartialEq<ExternalPid> for ExternalPid {
    fn eq(&self, other: &ExternalPid) -> bool {
        self.node == other.node && self.pid == other.pid
    }
}

impl PartialOrd<ExternalPid> for ExternalPid {
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        use cmp::Ordering;
        match self.node.partial_cmp(&other.node) {
            Some(Ordering::Equal) => self.pid.partial_cmp(&other.pid),
            result => result,
        }
    }
}
impl fmt::Debug for ExternalPid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalPid")
            .field("header", &self.header.as_usize())
            .field("node", &self.node)
            .field("next", &self.next)
            .field("pid", &self.pid)
            .finish()
    }
}

impl TryFrom<Term> for Pid {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Pid {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Pid(pid) => Ok(pid),
            _ => Err(TypeError),
        }
    }
}

#[derive(Debug)]
pub enum OutOfRange {
    Number,
    Serial,
}

struct Counter {
    serial: usize,
    number: usize,
}

impl Default for Counter {
    fn default() -> Counter {
        Counter {
            serial: 0,
            number: 0,
        }
    }
}

impl Counter {
    pub fn next(&mut self) -> Pid {
        let local_pid = unsafe { Pid::new_unchecked(self.number, self.serial) };

        if Pid::NUMBER_MAX <= self.number {
            self.serial += 1;
            self.number = 0;

            assert!(self.serial <= Pid::SERIAL_MAX)
        } else {
            self.number += 1;
        }

        local_pid
    }
}

lazy_static! {
    static ref RW_LOCK_COUNTER: RwLock<Counter> = Default::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    mod counter {
        use super::*;

        mod next {
            use super::*;

            #[test]
            fn number_rolling_over_increments_serial() {
                let mut counter: Counter = Default::default();
                let mut pid = counter.next();

                assert_eq!(pid, Term::local_pid(0, 0).unwrap());

                for _ in 0..NUMBER_MAX + 1 {
                    pid = counter.next();
                }

                assert_eq!(pid, Term::local_pid(0, 1).unwrap());
            }
        }
    }
}
