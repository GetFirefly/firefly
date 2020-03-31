use core::fmt::{self, Display};
use core::sync::atomic::{self, AtomicU32};

use lazy_static::lazy_static;

/// Generates the next `ID`.  `ID`s are not reused for the lifetime of the VM.
pub fn next() -> ID {
    let raw = ID_COUNTER.fetch_add(1, atomic::Ordering::AcqRel);

    ID(raw)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct ID(u32);

impl Display for ID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for ID {
    fn from(u: u32) -> Self {
        Self(u)
    }
}

impl Into<u32> for ID {
    fn into(self) -> u32 {
        self.0
    }
}

impl Into<u128> for ID {
    fn into(self) -> u128 {
        self.0 as u128
    }
}

lazy_static! {
    static ref ID_COUNTER: AtomicU32 = AtomicU32::new(0);
}
