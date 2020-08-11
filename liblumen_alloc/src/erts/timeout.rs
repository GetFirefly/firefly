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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiveTimeout {
    Immediate,
    Absolute(Monotonic),
    Infinity,
}
impl ReceiveTimeout {
    pub fn new(monotonic: Monotonic, timeout: Timeout) -> Self {
        match timeout {
            Timeout::Immediate => Self::Immediate,
            Timeout::Duration(milliseconds) => Self::Absolute(monotonic + milliseconds),
            Timeout::Infinity => Self::Infinity,
        }
    }

    pub fn is_timed_out(&self, time: Monotonic) -> bool {
        match self {
            Self::Immediate => true,
            Self::Absolute(end) => *end <= time,
            Self::Infinity => false,
        }
    }
}
impl Default for ReceiveTimeout {
    #[inline]
    fn default() -> Self {
        Self::Infinity
    }
}
