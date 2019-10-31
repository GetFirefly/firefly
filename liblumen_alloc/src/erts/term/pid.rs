use core::alloc::Layout;
use core::cmp;
use core::convert::TryFrom;
use core::default::Default;
use core::fmt::{self, Display};
use core::hash::{Hash, Hasher};
use core::ptr;

use liblumen_core::locks::RwLock;

use lazy_static::lazy_static;
use thiserror::Error;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::term::prelude::*;
use crate::erts::{HeapAlloc, Node};

lazy_static! {
    static ref RW_LOCK_COUNTER: RwLock<Counter> = Default::default();
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AnyPid {
    Local(Pid),
    External(Boxed<ExternalPid>),
}
impl Display for AnyPid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AnyPid::Local(pid) => write!(f, "{}", pid),
            AnyPid::External(pid) => write!(f, "{}", pid.as_ref()),
        }
    }
}
impl CloneToProcess for AnyPid {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        match self {
            AnyPid::Local(pid) => Ok(pid.encode().unwrap()),
            AnyPid::External(pid) => pid.clone_to_heap(heap),
        }
    }
}
impl TryFrom<TypedTerm> for AnyPid {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Pid(pid) => Ok(AnyPid::Local(pid)),
            TypedTerm::ExternalPid(pid) => Ok(AnyPid::External(pid)),
            _ => Err(TypeError),
        }
    }
}
impl From<Pid> for AnyPid {
    fn from(pid: Pid) -> Self {
        AnyPid::Local(pid)
    }
}
impl From<Boxed<ExternalPid>> for AnyPid {
    fn from(pid: Boxed<ExternalPid>) -> Self {
        AnyPid::External(pid)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Pid(usize);
impl Pid {
    const NUMBER_BIT_COUNT: u8 = 15;
    const NUMBER_MASK: usize = 0b111_1111_1111_1111;

    // The serial bit count is always 13 bits even though more could fit because they must be able
    // to fit in the PID_EXT and NEW_PID_EXT external term formats.
    const SERIAL_BIT_COUNT: u8 = 13;
    const SERIAL_MASK: usize = (0b1_1111_1111_1111) << Self::NUMBER_BIT_COUNT;

    pub const NUMBER_MAX: usize = (1 << (Self::NUMBER_BIT_COUNT as usize)) - 1;
    pub const SERIAL_MAX: usize = (1 << (Self::SERIAL_BIT_COUNT as usize)) - 1;
    pub const SIZE_IN_WORDS: usize = 1;

    /// Generates the next `Pid`.
    ///
    /// `Pid`s are not reused for the lifetime of the VM.
    pub fn next() -> Pid {
        let mut writable_counter = RW_LOCK_COUNTER.write();

        writable_counter.next()
    }

    /// Same as `next`, but directly encodes to `Term`
    pub fn next_term() -> Term {
        Self::next().encode().unwrap()
    }

    /// Given a the raw pid value (as a usize), reifies it into a `Pid`
    #[inline]
    pub unsafe fn from_raw(pid: usize) -> Self {
        Self(pid)
    }

    pub fn new(number: usize, serial: usize) -> Result<Pid, InvalidPidError> {
        Self::check(number, serial)
            .map(|(number, serial)| unsafe { Self::new_unchecked(number, serial) })
    }

    unsafe fn new_unchecked(number: usize, serial: usize) -> Pid {
        Self::from_raw((serial << (Self::NUMBER_BIT_COUNT as usize)) | number)
    }

    /// Same as `new`, but directly encodes to `Term`
    pub fn make_term(number: usize, serial: usize) -> Result<Term, InvalidPidError> {
        let pid = Self::new(number, serial)?;
        Ok(pid.encode().unwrap())
    }

    pub fn check(number: usize, serial: usize) -> Result<(usize, usize), InvalidPidError> {
        if number <= Self::NUMBER_MAX {
            if serial <= Self::SERIAL_MAX {
                Ok((number, serial))
            } else {
                Err(InvalidPidError::Serial)
            }
        } else {
            Err(InvalidPidError::Number)
        }
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

    #[inline(always)]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl Display for Pid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#PID<0.{}.{}>", self.number(), self.serial())
    }
}

impl PartialEq<ExternalPid> for Pid {
    #[inline(always)]
    fn eq(&self, _other: &ExternalPid) -> bool {
        false
    }
}
impl<T> PartialEq<Boxed<T>> for Pid
where
    T: PartialEq<Pid>,
{
    #[inline]
    default fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}
impl PartialEq<Boxed<ExternalPid>> for Pid {
    #[inline(always)]
    fn eq(&self, _other: &Boxed<ExternalPid>) -> bool {
        false
    }
}
impl PartialOrd<ExternalPid> for Pid {
    #[inline]
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        self.partial_cmp(&other.pid)
    }
}
impl<T> PartialOrd<Boxed<T>> for Pid
where
    T: PartialOrd<Pid>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
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
#[repr(C)]
pub struct ExternalPid {
    header: Header<ExternalPid>,
    node: Node,
    next: *mut u8, // off heap header
    pid: Pid,
}
impl ExternalPid {
    pub(in crate::erts) fn with_node_id(
        node_id: usize,
        number: usize,
        serial: usize,
    ) -> Result<Self, InvalidPidError> {
        let node = Node::new(node_id);

        Self::new(node, number, serial)
    }

    fn new(node: Node, number: usize, serial: usize) -> Result<Self, InvalidPidError> {
        let pid = Pid::new(number, serial)?;

        Ok(Self {
            header: Default::default(),
            node,
            next: ptr::null_mut(),
            pid,
        })
    }
}

impl CloneToProcess for ExternalPid {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + HeapAlloc,
    {
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, 1);

            Ok(ptr.into())
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

impl Hash for ExternalPid {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.pid.hash(state);
    }
}

impl Eq for ExternalPid {}
impl PartialEq for ExternalPid {
    fn eq(&self, other: &ExternalPid) -> bool {
        self.node == other.node && self.pid == other.pid
    }
}
impl<T> PartialEq<Boxed<T>> for ExternalPid
where
    T: PartialEq<ExternalPid>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl Ord for ExternalPid {
    fn cmp(&self, other: &ExternalPid) -> cmp::Ordering {
        self.node
            .cmp(&other.node)
            .then_with(|| self.pid.cmp(&other.pid))
    }
}
impl PartialOrd for ExternalPid {
    fn partial_cmp(&self, other: &ExternalPid) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> PartialOrd<Boxed<T>> for ExternalPid
where
    T: PartialOrd<ExternalPid>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<cmp::Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}
impl TryFrom<TypedTerm> for Boxed<ExternalPid> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::ExternalPid(pid) => Ok(pid),
            _ => Err(TypeError),
        }
    }
}

#[derive(Error, Debug)]
pub enum InvalidPidError {
    #[error("invalid pid: number out of range")]
    Number,
    #[error("invalid pid: serial out of range")]
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
