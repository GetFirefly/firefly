use core::time::Duration;
use thiserror::Error;

use super::time::Monotonic;

#[derive(Debug, Error)]
#[error("invalid timeout value, must be `infinity` or a non-negative integer value")]
pub struct InvalidTimeoutError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timeout {
    Infinity,
    Immediate,
    Duration(Duration),
}
impl Timeout {
    pub fn from_millis<T: Into<isize>>(to: T) -> Result<Self, InvalidTimeoutError> {
        match to.into() {
            0 => Ok(Self::Immediate),
            ms if ms >= 0 => Ok(Self::Duration(Duration::from_millis(ms as u64))),
            _ => Err(InvalidTimeoutError),
        }
    }

    pub fn as_duration(&self) -> Option<Duration> {
        match self {
            Self::Infinity => None,
            Self::Immediate => Some(Duration::from_millis(0)),
            Self::Duration(ref d) => Some(d.clone()),
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
    Infinity,
    Immediate,
    Duration { start: Duration, duration: Duration },
}
impl ReceiveTimeout {
    pub fn new(monotonic: Monotonic, timeout: Timeout) -> Self {
        match timeout {
            Timeout::Infinity => Self::Infinity,
            Timeout::Immediate => Self::Immediate,
            Timeout::Duration(duration) => {
                let start = Duration::from_millis(monotonic.0);
                Self::Duration { start, duration }
            }
        }
    }

    pub fn is_timed_out(&self, monotonic: Monotonic) -> bool {
        match self {
            Self::Infinity => false,
            Self::Immediate => true,
            Self::Duration { start, duration } => {
                let end = Duration::from_millis(monotonic.0);
                end - *start >= *duration
            }
        }
    }
}
impl Default for ReceiveTimeout {
    #[inline]
    fn default() -> Self {
        Self::Infinity
    }
}
