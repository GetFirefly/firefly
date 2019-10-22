use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::default::Default;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::ptr;

use liblumen_core::locks::RwLock;

use lazy_static::lazy_static;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::term::{arity_of, AsTerm, Term, TypeError, TypedTerm};
use crate::erts::{HeapAlloc, Node};

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

    /// Never exceeds 15 significant bits to remain compatible with `PID_EXT` and
    /// `NEW_PID_EXT` external term formats.
    pub fn number(&self) -> u16 {
        (self.0 & Self::NUMBER_MASK) as u16
    }

    /// Never exceeds 15 significant bits to remain compatible with `PID_EXT` and `NEW_PID_EXT`
    /// external term formats.
    pub fn serial(&self) -> u16 {
        ((self.0 & Self::SERIAL_MASK) >> Self::NUMBER_BIT_COUNT) as u16
    }

    const NUMBER_BIT_COUNT: u8 = 15;
    const NUMBER_MASK: usize = 0b111_1111_1111_1111;
    pub const NUMBER_MAX: usize = (1 << (Self::NUMBER_BIT_COUNT as usize)) - 1;

    // The serial bit count is always 13 bits even though more could fit because they must be able
    // to fit in the PID_EXT and NEW_PID_EXT external term formats.
    const SERIAL_BIT_COUNT: u8 = 13;
    const SERIAL_MASK: usize = (0b1_1111_1111_1111) << Self::NUMBER_BIT_COUNT;
    pub const SERIAL_MAX: usize = (1 << (Self::SERIAL_BIT_COUNT as usize)) - 1;

    pub const SIZE_IN_WORDS: usize = 1;
}

unsafe impl AsTerm for Pid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_pid(self.0)
    }
}

impl Display for Pid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#PID<0.{}.{}>", self.number(), self.serial())
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
impl ExternalPid {
    pub(in crate::erts) fn with_node_id(
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Self, OutOfRange> {
        let node = Node::new(node_id);

        Self::new(node, number, serial)
    }

    fn new(node: Node, number: usize, serial: usize) -> Result<Self, OutOfRange> {
        let pid = Pid::new(number, serial)?;
        let header = Term::make_header(arity_of::<Self>(), Term::FLAG_EXTERN_PID);

        Ok(Self {
            header,
            node,
            next: ptr::null_mut(),
            pid,
        })
    }
}

unsafe impl AsTerm for ExternalPid {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for ExternalPid {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        unsafe {
            let ptr = heap.alloc(self.size_in_words())?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, 1);

            Ok(Term::make_boxed(ptr))
        }
    }
}

impl Display for ExternalPid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "<{}.{}.{}>",
            self.node.id(),
            self.pid.number(),
            self.pid.serial()
        )
    }
}

impl Eq for ExternalPid {}

impl Hash for ExternalPid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.pid.hash(state);
    }
}

impl Ord for ExternalPid {
    fn cmp(&self, other: &ExternalPid) -> cmp::Ordering {
        self.node
            .cmp(&other.node)
            .then_with(|| self.pid.cmp(&other.pid))
    }
}

impl PartialEq<ExternalPid> for ExternalPid {
    fn eq(&self, other: &ExternalPid) -> bool {
        self.node == other.node && self.pid == other.pid
    }
}

impl PartialOrd<ExternalPid> for ExternalPid {
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl fmt::Debug for ExternalPid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ExternalPid")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
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

                assert_eq!(pid, Pid::new(0, 0).unwrap());

                for _ in 0..Pid::NUMBER_MAX + 1 {
                    pid = counter.next();
                }

                assert_eq!(pid, Pid::new(0, 1).unwrap());
            }
        }
    }
}
