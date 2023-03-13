use core::mem;
use core::num::NonZeroUsize;

use crate::term::{atoms, OpaqueTerm, Term};

/// Represents what priority queue a given process resides in
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 1 << 1,
    Max = 1 << 2,
}
impl Into<OpaqueTerm> for Priority {
    fn into(self) -> OpaqueTerm {
        match self {
            Self::Low => atoms::Low.into(),
            Self::Normal => atoms::Normal.into(),
            Self::High => atoms::High.into(),
            Self::Max => atoms::Max.into(),
        }
    }
}
impl TryFrom<Term> for Priority {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Atom(a) => match a {
                a if a == atoms::Low => Ok(Self::Low),
                a if a == atoms::Normal => Ok(Self::Normal),
                a if a == atoms::High => Ok(Self::High),
                a if a == atoms::Max => Ok(Self::Max),
                _ => Err(()),
            },
            _ => Err(()),
        }
    }
}

/// This represents the current `max_heap_size` configuration of a process
#[derive(Debug, Copy, Clone)]
pub struct MaxHeapSize {
    pub size: Option<NonZeroUsize>,
    pub kill: bool,
    pub error_logger: bool,
}
impl MaxHeapSize {
    const KILL_BIT: usize = (mem::size_of::<usize>() * 8) - 1;
    const LOGGER_BIT: usize = (mem::size_of::<usize>() * 8) - 2;
    const SIZE_BITS: usize = (mem::size_of::<usize>() * 8) - 2;
    const MAX_SIZE: usize = (1 << Self::SIZE_BITS) - 1;

    pub fn kill(mut self, enabled: bool) -> Self {
        self.kill = enabled;
        self
    }

    pub fn error_logger(mut self, enabled: bool) -> Self {
        self.error_logger = enabled;
        self
    }

    pub fn set(mut self, size: usize) -> Self {
        self.size = NonZeroUsize::new(size);
        self
    }
}
impl Default for MaxHeapSize {
    fn default() -> Self {
        Self {
            size: None,
            kill: true,
            error_logger: true,
        }
    }
}
impl TryFrom<Term> for MaxHeapSize {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Int(i) if i >= 0 => Ok(Self {
                size: NonZeroUsize::new(i.try_into().map_err(|_| ())?),
                kill: true,
                error_logger: true,
            }),
            Term::Map(opts) => {
                let mut max_heap_size = Self::default();
                if let Some(v) = opts.get(atoms::Size) {
                    match v.into() {
                        Term::Int(i) if i >= 0 => {
                            max_heap_size.size = NonZeroUsize::new(i.try_into().map_err(|_| ())?);
                        }
                        _ => return Err(()),
                    }
                }
                if let Some(v) = opts.get(atoms::Kill) {
                    max_heap_size.kill = match v {
                        OpaqueTerm::TRUE => true,
                        OpaqueTerm::FALSE => false,
                        _ => return Err(()),
                    };
                }
                if let Some(v) = opts.get(atoms::ErrorLogger) {
                    max_heap_size.error_logger = match v {
                        OpaqueTerm::TRUE => true,
                        OpaqueTerm::FALSE => false,
                        _ => return Err(()),
                    };
                }
                Ok(max_heap_size)
            }
            _ => Err(()),
        }
    }
}
impl firefly_system::sync::Atom for MaxHeapSize {
    type Repr = usize;

    #[inline]
    fn pack(self) -> Self::Repr {
        let mut raw = self.size.map(|nz| nz.get()).unwrap_or(0);
        assert!(raw <= Self::MAX_SIZE);
        if self.kill {
            raw |= 1 << Self::KILL_BIT;
        }
        if self.error_logger {
            raw |= 1 << Self::LOGGER_BIT;
        }
        raw
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        let size = NonZeroUsize::new(raw & Self::MAX_SIZE);
        let kill = raw & Self::KILL_BIT == Self::KILL_BIT;
        let error_logger = raw & Self::LOGGER_BIT == Self::LOGGER_BIT;
        Self {
            size,
            kill,
            error_logger,
        }
    }
}

bitflags::bitflags! {
    /// These flags are private to a process and require holding a [`ProcessLock`] to read/modify
    pub struct ProcessFlags: u32 {
        /// Schedule out this process after a hibernate op
        const HIBERNATE = 1;
        /// This process is in the timer queue
        const IN_TIMER_QUEUE = 1 << 1;
        /// The process has timed out on an operation (like a receive)
        const TIMEOUT = 1 << 2;
        /// The next GC should force grow the heap
        const HEAP_GROW = 1 << 3;
        /// The next GC must be a full sweep
        const NEED_FULLSWEEP = 1 << 4;
        /// This process owns ETS tables
        const USING_DB = 1 << 5;
        /// This process is used in distribution
        const DISTRIBUTION = 1 << 6;
        /// This process is using the DDLL interface
        const USING_DDLL = 1 << 7;
        /// This process is an ETS super user
        const ETS_SUPER_USER = 1 << 8;
        /// Force garbage collection the next time this process is scheduled
        const FORCE_GC = 1 << 9;
        /// Disable garbage collection, for use by BIFs that yield and need to prevent GC
        const DISABLE_GC = 1 << 10;
        /// Process has usage of abandoned heap
        const ABANDONED_HEAP_USE = 1 << 11;
        /// Delays garbage collection until just prior to scheduling out the process
        ///
        /// A process must never be scheduled out while this is set
        const DELAY_GC = 1 << 12;
        /// Dirty GC hibernate scheduled
        const DIRTY_GC_HIBERNATE = 1 << 13;
        /// Dirty major GC scheduled
        const DIRTY_MAJOR_GC = 1 << 14;
        /// Dirty minor GC scheduled
        const DIRTY_MINOR_GC = 1 << 15;
        /// Process is hibernated
        const HIBERNATED = 1 << 16;
        /// Process is trapping exits
        const TRAP_EXIT = 1 << 17;
        /// Process is doing a distributed fragmented send
        const FRAGMENTED_SEND = 1 << 18;

        /// An alias for the GC indicator flags
        ///
        /// This does not include the DISABLE_GC or DELAY_GC flags, which are for a different purpose
        const GC_FLAGS = Self::HEAP_GROW.bits | Self::NEED_FULLSWEEP.bits | Self::FORCE_GC.bits;
    }
}

bitflags::bitflags! {
    /// These flags are public and may be modified at any time in multiple threads.
    ///
    /// Some of these flags are implicitly owned by the holder of a [`ProcessLock`], such
    /// as the priority flags, and the EXITING, FREE, RUNNING and GC statuses.
    pub struct StatusFlags: u32 {
        /// Process is enqueued in the low priority queue
        const PRIORITY_LOW = 1 << 1;
        /// Process is enqueued in the normal priority queue
        const PRIORITY_NORMAL = 1 << 2;
        /// Process is enqueued in the high priority queue
        const PRIORITY_HIGH = 1 << 3;
        /// Process is enqueued in the max priority queue
        const PRIORITY_MAX = 1 << 4;
        /// Process is exiting, but not registered in process registry any more
        const FREE = 1 << 5;
        /// Process is exiting, but still visible in process registry
        const EXITING = 1 << 6;
        /// Process wants to execute
        const ACTIVE = 1 << 7;
        /// Process wants to handle system tasks/signals
        const ACTIVE_SYS = 1 << 8;
        /// Process is scheduled in a run queue
        const SCHEDULED = 1 << 9;
        /// Process is currently executing
        const RUNNING = 1 << 10;
        /// Process is currently executing system tasks/signals
        const RUNNING_SYS = 1 << 11;
        /// Process is currently suspended, suppresses ACTIVE, but not ACTIVE_SYS
        const SUSPENDED = 1 << 12;
        /// Process is currently garbage collecting
        const GC = 1 << 13;
        /// Indicates that this process has normal system tasks scheduled
        const SYS_TASKS = 1 << 14;
        /// We may have outstanding signals from self()
        const MAYBE_SELF_SIGNALS = 1 << 15;
        /// Process has unhandled signals in its in-transit signal queue
        const HAS_IN_TRANSIT_SIGNALS = 1 << 16;
        /// Process has unhandled signals in its received queue
        const HAS_PENDING_SIGNALS = 1 << 17;
        /// Process has an off-heap message queue (currently the default)
        const OFF_HEAP_MSGQ = 1 << 18;
        /// Process has sensitive data, so disable certain introspection features
        const SENSITIVE = 1 << 19;

        /// The default process flags
        const DEFAULT = Self::PRIORITY_NORMAL.bits | Self::OFF_HEAP_MSGQ.bits;

        /// A mask of all the priority bits, used for converting their values into a [`Priority`] enum
        const PRIORITY_MASK = Self::PRIORITY_LOW.bits
            | Self::PRIORITY_NORMAL.bits
            | Self::PRIORITY_HIGH.bits
            | Self::PRIORITY_MAX.bits;

        /// A mask of all the primary status bits.
        ///
        /// If none of these are set, a process is considered runnable
        const STATUS_MASK = Self::FREE.bits
            | Self::EXITING.bits
            | Self::ACTIVE.bits
            | Self::ACTIVE_SYS.bits
            | Self::SCHEDULED.bits
            | Self::RUNNING.bits
            | Self::RUNNING_SYS.bits
            | Self::SUSPENDED.bits
            | Self::GC.bits;
    }
}
impl Default for StatusFlags {
    #[inline]
    fn default() -> Self {
        Self::DEFAULT
    }
}
impl StatusFlags {
    /// Returns the current process priority
    pub fn priority(&self) -> Priority {
        unsafe {
            core::mem::transmute::<u8, Priority>((self.bits() & Self::PRIORITY_MASK.bits()) as u8)
        }
    }

    /// Returns true if this process is ready to be picked up by a scheduler and run
    pub fn is_runnable(&self) -> bool {
        !self.contains(Self::STATUS_MASK)
    }

    /// Returns true if this process is currently executing
    pub fn is_running(&self) -> bool {
        self.contains(Self::RUNNING | Self::RUNNING_SYS)
    }

    /// Returns true if this process has any unhandled signals pending
    pub fn has_signals(&self) -> bool {
        self.contains(
            Self::MAYBE_SELF_SIGNALS | Self::HAS_IN_TRANSIT_SIGNALS | Self::HAS_PENDING_SIGNALS,
        )
    }
}
impl core::ops::BitOr<Priority> for StatusFlags {
    type Output = StatusFlags;

    fn bitor(mut self, rhs: Priority) -> Self::Output {
        self |= rhs;
        self
    }
}
impl core::ops::BitOrAssign<Priority> for StatusFlags {
    fn bitor_assign(&mut self, rhs: Priority) {
        self.remove(Self::PRIORITY_MASK);
        *self |= match rhs {
            Priority::Low => Self::PRIORITY_LOW,
            Priority::Normal => Self::PRIORITY_NORMAL,
            Priority::High => Self::PRIORITY_HIGH,
            Priority::Max => Self::PRIORITY_MAX,
        };
    }
}
impl firefly_system::sync::Atom for StatusFlags {
    type Repr = u32;

    #[inline]
    fn pack(self) -> Self::Repr {
        self.bits()
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        unsafe { StatusFlags::from_bits_unchecked(raw) }
    }
}
impl firefly_system::sync::AtomLogic for StatusFlags {}
