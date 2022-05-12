use alloc::sync::Arc;
use core::any::TypeId;
use core::fmt::{self, Display};

use anyhow::anyhow;

use super::{Node, Term};

/// This struct abstracts over the locality of a process identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pid {
    Local { id: ProcessId }
    External { id: ProcessId, node: Arc<Node> }
}
impl Pid {
    pub const TYPE_ID: TypeId = TypeId::of::<Pid>();

    /// Creates a new local pid, manually.
    ///
    /// This function will return an error if the number/serial components are out of range.
    pub fn new_local(number: usize, serial: usize) -> anyhow::Result<Self> {
        let id = ProcessId::new(number, serial)?;
        Ok(Self::Local { id })
    }

    /// Creates a new external pid, manually.
    ///
    /// This function will return an error if the number/serial components are out of range.
    pub fn new_external(node: Arc<Node>, number: usize, serial: usize) -> anyhow::Result<Self> {
        let id = ProcessId::new(number, serial)?;
        Ok(Self::External { id, node })
    }

    /// Allocates a new local pid, using the global counter.
    ///
    /// NOTE: The pid returned by this function is not guaranteed to be unique. Once the pid
    /// space has been exhausted at least once, pids may be reused, and it is up to the caller
    /// to ensure that a pid is only used by a single live process on the local node at any given
    /// time.
    #[inline]
    pub fn next() -> Self {
        Self::Local { id: ProcessId::next() }
    }

    /// Returns the raw process identifier
    pub fn id(&self) -> ProcessId {
        match self {
            Self::Local { id }
            | Self::External { id, .. } => *id,
        }
    }

    /// Returns the node associated with this pid, if applicable
    pub fn node(&self) -> Option<Arc<Node>> {
        match self {
            Self::External { node, .. } => Some(node.clone()),
            _ => None,
        }
    }
}
impl TryFrom<Term> for Pid {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Pid(pid) => Ok(Pid::clone(&pid)),
            _ => Err(()),
        }
    }
}
impl Display for Pid {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Local { id } => write!(f, "<0.{}.{}>", id.number(), id.serial()),
            Self::External { id, node } => write!(f, "<{}.{}.{}>", node.id(), id.number(), id.serial()),
        }
    }
}
impl Ord for Pid {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match (self, other) {
            (Self::Local { id: x }, Self::Local { id: y }) => x.cmp(y),
            (Self::Local { .. }, _) => Ordering::Less,
            (Self::External { id: xid, node: xnode }, Self::External { id: yid, node: ynode }) => {
                match xnode.cmp(ynode) {
                    Ordering::Equal => xid.cmp(yid),
                    other => other,
                }
            }
        }
    }
}
impl PartialOrd for Pid {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// This struct represents a process identifier, without the node component.
///
/// Local pids in the BEAM are designed to fit in an immediate, which limits their range to a value
/// that can be expressed in a single 32-bit word. However, because external pids are always 64 bits,
/// (both number and serial are given a full 32-bits), we choose to use 64-bits in both cases, storing
/// the serial in the high 32-bits, and the number in the low 32 bits. We do still impose a restriction
/// on the maximum value of local pids, for both number and serial, that is less than 32-bits, allowing
/// us to detect overflow when calculating the next possibly-available id.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ProcessId(u64);
impl ProcessId {
    // We limit the range to 31 bits for both numbers and serials
    const NUMBER_MAX: u64 = (1 << 31) - 1;
    const SERIAL_MAX: u64 = Self::NUMBER_MAX << 32;
    const NUMBER_MASK: u64 = (-1i64 as u64) >> 32;
    const SERIAL_MASK: u64 = !Self::NUMBER_MASK;

    /// Returns the number component of this process identifier
    ///
    /// NOTE: The value returned is guaranteed to never exceed 31 significant bits, so
    /// as to remain compatible with External Term Format.
    pub fn number(&self) -> u32 {
        (self.0 & Self::NUMBER_MASK) as u32
    }

    /// Returns the serial component of this process identifier
    ///
    /// NOTE: The value returned is guaranteed to never exceed 31 significant bits, so
    /// as to remain compatible with External Term Format.
    pub fn serial(&self) -> u32 {
        (self.0 & Self::SERIAL_MASK >> 32) as u32
    }

    #[inline(always)]
    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }

    /// Creates a process identifier from the given number and serial components, manually.
    ///
    /// This function will return `Err` if either component is out of range.
    pub fn new(number: usize, serial: usize) -> anyhow::Result<Self> {
        let number = number as u64;
        let serial = serial as u64;
        if serial > Self::SERIAL_MAX {
            return Err(anyhow!("invalid pid, serial is too large"));
        }
        if number > Self::NUMBER_MAX {
            return Err(anyhow!("invalid pid, number is too large"));
        }
        Ok(unsafe { Self::new_unchecked(number, serial) })
    }

    /// Creates a process identifier from the given number and serial components, bypassing
    /// the normal validation checks.
    ///
    /// # Safety
    ///
    /// This function should only be called in contexts where the number/serial have already been
    /// validated or are guaranteed to be valid. Callers must always uphold the guarantees provided
    /// by this module (i.e. in terms of the valid range of the number and serial components).
    pub unsafe fn new_unchecked(number: u64, serial: u64) -> Self {
        debug_assert!(serial <= Self::SERIAL_MAX, "invalid pid, serial is too large");
        debug_assert!(number <= Self::NUMBER_MAX, "invalid pid, number is too large");

        Self::from_raw((serial << 32) | number)
    }

    /// Generates the next process id.
    ///
    /// # Safety
    ///
    /// Process identifiers can roll over eventually, assuming the program runs long enough and
    /// that processes are being regularly spawned. Once the pid space has been exhausted, calling this
    /// function will produce a pid that starts over at its initial state. It is required that the scheduler
    /// verify that a pid is unused by any live process before spawning a process with a pid returned by this
    /// function.
    pub fn next() -> Self {
        use core::sync::atomic::{AtomicU64, Ordering::SeqCst};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        Self(COUNTER.fetch_update(SeqCst, SeqCst, calculate_next_pid).unwrap())
    }

    /// Given a the raw process id value (as a usize), reifies it into a `ProcessId`
    #[inline]
    pub unsafe const fn from_raw(pid: u64) -> Self {
        debug_assert!(pid & Self::SERIAL_MASK <= Self::SERIAL_MAX, "invalid pid, serial is too large");
        debug_assert!(pid & Self::NUMBER_MASK <= Self::NUMBER_MAX, "invalid pid, number is too large");
        Self(pid)
    }
}

// This has been extracted from ProcessId::next() to allow for testing various scenarios without
// polluting the global pid counter.
fn calculate_next_pid(x: u64) -> Option<u64> {
    use core::intrinsics::likely;
    // The layout in memory of a process identifier is as follows:
    //
    //     000SSSSSSSSSSSSS0NNNNNNNNNNNNNNN
    //
    //     0 = unused
    //     S = serial bit
    //     N = number bit
    //
    // The serial/number ranges are limited to these sizes because they must be able to fit
    // in the PID_EXT and NEW_PID_EXT external term formats, so even though we could support
    // arbitrarily large pids in theory, we can't in practice.
    const NUMBER_INC: u64 = 1;
    const SERIAL_INC: u64 = 1 << 16;

    // Fast path
    if unsafe { likely(x & ProcessId::NUMBER_MASK < ProcessId::NUMBER_MAX) } {
        return Some(x + NUMBER_INC);
    }

    // Slow path
    let next = (x & ProcessId::SERIAL_MASK) + SERIAL_INC;
    if unsafe { likely(next & ProcessId::SERIAL_MASK <= ProcessId::SERIAL_MAX) } {
        return Some(next);
    }

    // Edge case for pid rollover
    Some(0)
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicU64, Ordering::SeqCst};

    use super::*;

    #[test]
    fn pid_rollover() {
        const MAX_PID: u64 = ProcessId::SERIAL_MAX | ProcessId::NUMBER_MAX;
        static SERIAL_ROLLOVER: AtomicU64 = AtomicU64::new(MAX_PID);
        let next = SERIAL_ROLLOVER.fetch_update(SeqCst, SeqCst, calculate_next_pid).unwrap();
        assert_eq!(next.as_u64(), MAX_PID);
        let next = SERIAL_ROLLOVER.fetch_update(SeqCst, SeqCst, calculate_next_pid).unwrap();
        assert_eq!(next.as_u64(), 0);
    }

    #[test]
    fn pid_serial_increment() {
        const MAX_NUM: u64 = ProcessId::NUMBER_MAX;
        static NUM_ROLLOVER: AtomicU64 = AtomicU64::new(MAX_NUM);
        let next = NUM_ROLLOVER.fetch_update(SeqCst, SeqCst, calculate_next_pid).unwrap();
        assert_eq!(next.as_u64(), MAX_NUM);
        let next = NUM_ROLLOVER.fetch_update(SeqCst, SeqCst, calculate_next_pid).unwrap();
        assert_eq!(next.as_u64(), (1u64 << 32));
    }
}
