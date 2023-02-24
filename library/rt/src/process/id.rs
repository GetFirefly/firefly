use core::fmt;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProcessIdError {
    SerialTooLarge,
    NumberTooLarge,
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
impl fmt::Display for ProcessId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<X.{}.{}>", self.number(), self.serial())
    }
}
impl ProcessId {
    // We limit the range to 32 bits for both numbers and serials
    const NUMBER_MAX: u64 = u32::MAX as u64;
    const SERIAL_MAX: u64 = Self::NUMBER_MAX;
    const SERIAL_MASK: u64 = u32::MAX as u64;

    /// Return the raw representation of this ProcessId as a u64 value
    #[inline(always)]
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Returns the number component of this process identifier
    ///
    /// NOTE: The value returned is guaranteed to never exceed 31 significant bits, so
    /// as to remain compatible with External Term Format.
    pub fn number(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Returns the serial component of this process identifier
    ///
    /// NOTE: The value returned is guaranteed to never exceed 31 significant bits, so
    /// as to remain compatible with External Term Format.
    pub fn serial(&self) -> u32 {
        (self.0 & Self::SERIAL_MASK) as u32
    }

    /// Creates a process identifier from the given number and serial components, manually.
    ///
    /// This function will return `Err` if either component is out of range.
    pub fn new(number: u32, serial: u32) -> Result<Self, ProcessIdError> {
        let n = number as u64;
        let s = serial as u64;
        if n > Self::NUMBER_MAX {
            return Err(ProcessIdError::NumberTooLarge);
        }
        if s > Self::SERIAL_MAX {
            return Err(ProcessIdError::SerialTooLarge);
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
    pub unsafe fn new_unchecked(number: u32, serial: u32) -> Self {
        let number = number as u64;
        let serial = serial as u64;
        debug_assert!(
            serial <= Self::SERIAL_MAX,
            "invalid pid, serial is too large"
        );
        debug_assert!(
            number <= Self::NUMBER_MAX,
            "invalid pid, number is too large"
        );

        Self::from_raw((number << 32) | serial)
    }

    /// Generates the next globally-unique process number.
    ///
    /// # SAFETY
    ///
    /// The maximum number of processes that can be spawned before we exhaust available
    /// process identifiers is 2^64 - sufficient to last a very very long time. However, we
    /// do protect against rollover, by guaranteeing that the first pid starts at 1, and
    /// asserting if we try to assign a pid of 0
    pub fn next() -> Self {
        use core::sync::atomic::{AtomicU64, Ordering};

        static COUNTER: AtomicU64 = AtomicU64::new(1);

        let next = COUNTER.fetch_add(1, Ordering::SeqCst);

        assert_ne!(
            next, 0,
            "system limit: exhausted all available process identifiers"
        );
        unsafe { Self::from_raw(next.rotate_left(32)) }
    }

    /// Given a the raw process id value (as a usize), reifies it into a `ProcessId`
    #[inline]
    pub unsafe fn from_raw(pid: u64) -> Self {
        debug_assert!(
            (pid >> 32) <= Self::NUMBER_MAX,
            "invalid pid, number is too large"
        );
        debug_assert!(
            pid & Self::SERIAL_MASK <= Self::SERIAL_MAX,
            "invalid pid, serial is too large"
        );
        Self(pid)
    }
}
