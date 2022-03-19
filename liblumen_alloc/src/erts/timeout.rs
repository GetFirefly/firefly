use thiserror::Error;

use super::time::{Milliseconds, Monotonic};

#[derive(Debug, Error)]
#[error("invalid timeout value, must be `infinity` or a non-negative integer value")]
pub struct InvalidTimeoutError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeout {
    Immediate,
    Duration(Milliseconds),
    Infinity,
}
impl Timeout {
    pub fn from_millis<T: Into<isize>>(to: T) -> Result<Self, InvalidTimeoutError> {
        match to.into() {
            0 => Ok(Self::Immediate),
            ms if ms >= 0 => Ok(Self::Duration(Milliseconds(ms as u64))),
            _ => Err(InvalidTimeoutError),
        }
    }
}
impl Default for Timeout {
    #[inline]
    fn default() -> Self {
        Self::Infinity
    }
}

/// This is a more compact encoding of Timeout
///
/// * A value of 0 is Immediate
/// * A value of u64::MAX is infinity
/// * Any other value is the monotonic clock instant at which the timeout occurs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ReceiveTimeout(u64);
impl ReceiveTimeout {
    const IMMEDIATE: Self = Self(0);
    const INFINITY: Self = Self(u64::MAX);

    /// Creates a new absolute timeout from a monotonic clock
    pub const fn absolute(value: Monotonic) -> Self {
        Self(value.0)
    }

    /// Creates a new receive timeout based on the current monotonic clock and the given timeout
    pub fn new(monotonic: Monotonic, timeout: Timeout) -> Self {
        match timeout {
            Timeout::Immediate => Self::IMMEDIATE,
            Timeout::Infinity => Self::INFINITY,
            Timeout::Duration(ms) => Self(monotonic.0 + ms.0),
        }
    }

    /// Returns this timeout as a monotonic clock value, if applicable
    pub const fn monotonic(self) -> Option<Monotonic> {
        match self {
            Self::IMMEDIATE | Self::INFINITY => None,
            Self(value) => Some(Monotonic(value)),
        }
    }

    /// Returns true if this timeout should be considered expired relative to the given monotonic clock time
    pub fn is_timed_out(self, time: Monotonic) -> bool {
        match self {
            Self::IMMEDIATE => true,
            Self::INFINITY => false,
            Self(value) => value <= time.0,
        }
    }
}
impl Default for ReceiveTimeout {
    #[inline(always)]
    fn default() -> Self {
        Self::INFINITY
    }
}
