use core::fmt::{self, Display};
use core::ops::{Add, Div, Mul, Rem, Sub};
use core::time::Duration;

use num_bigint::BigInt;

// Must be at least a `u64` because `u32` is only ~49 days (`(1 << 32)`)
/// A duration in milliseconds between `Monotonic` times.
#[derive(Clone, Copy, Eq, Debug, PartialEq, PartialOrd)]
pub struct Milliseconds(pub u64);

impl Milliseconds {
    pub const fn const_div(self, rhs: u64) -> Self {
        Self(self.0 / rhs)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl Add<Milliseconds> for Milliseconds {
    type Output = Milliseconds;

    fn add(self, rhs: Milliseconds) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Div<u64> for Milliseconds {
    type Output = Milliseconds;

    fn div(self, rhs: u64) -> Self::Output {
        self.const_div(rhs)
    }
}

impl Mul<u64> for Milliseconds {
    type Output = Milliseconds;

    fn mul(self, rhs: u64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl From<Milliseconds> for BigInt {
    fn from(milliseconds: Milliseconds) -> Self {
        milliseconds.0.into()
    }
}

impl From<Milliseconds> for u64 {
    fn from(milliseconds: Milliseconds) -> Self {
        milliseconds.0
    }
}

impl From<Monotonic> for Milliseconds {
    fn from(monotonic: Monotonic) -> Self {
        Self(monotonic.0)
    }
}

/// The absolute time
#[derive(Clone, Copy, Eq, Debug, Ord, PartialEq, PartialOrd)]
pub struct Monotonic(pub u64);

impl Monotonic {
    pub fn from_millis<T: Into<u64>>(to: T) -> Self {
        Self(to.into())
    }

    pub fn checked_sub(&self, rhs: Self) -> Option<Milliseconds> {
        self.0.checked_sub(rhs.0).map(Milliseconds)
    }

    pub fn round_down(&self, divisor: u64) -> Self {
        Self((self.0 / divisor) * divisor)
    }
}

impl Add<Duration> for Monotonic {
    type Output = Monotonic;

    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + (rhs.as_millis() as u64))
    }
}

impl Add<Milliseconds> for Monotonic {
    type Output = Monotonic;

    fn add(self, rhs: Milliseconds) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Display for Monotonic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ms", self.0)
    }
}

impl Rem<Milliseconds> for Monotonic {
    type Output = Milliseconds;

    fn rem(self, rhs: Milliseconds) -> Self::Output {
        Milliseconds(self.0 % rhs.0)
    }
}

impl Sub<Milliseconds> for Monotonic {
    type Output = Monotonic;

    fn sub(self, rhs: Milliseconds) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub<Monotonic> for Monotonic {
    type Output = Milliseconds;

    fn sub(self, rhs: Monotonic) -> Self::Output {
        Milliseconds(self.0 - rhs.0)
    }
}
