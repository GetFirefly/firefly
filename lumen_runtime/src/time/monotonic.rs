use num_bigint::BigInt;

use liblumen_core::locks::RwLock;

use crate::time::convert;
use crate::time::Unit::{self, *};

// Must be at least a `u64` because `u32` is only ~49 days (`(1 << 32)`)
pub type Milliseconds = u64;
pub type Source = fn() -> Milliseconds;

pub fn time(unit: Unit) -> BigInt {
    let milliseconds = time_in_milliseconds();

    match unit {
        Second => (milliseconds / MILLISECONDS_PER_SECOND).into(),
        Millisecond => milliseconds.into(),
        Microsecond => (milliseconds * MICROSECONDS_PER_MILLISECOND).into(),
        Nanosecond => (milliseconds * NANOSECONDS_PER_MILLISECONDS).into(),
        _ => convert(
            (milliseconds * NANOSECONDS_PER_MILLISECONDS).into(),
            Nanosecond,
            unit,
        ),
    }
}

pub fn time_in_milliseconds() -> Milliseconds {
    (*RW_LOCK_SOURCE.read())()
}

pub fn set_source(source: Source) {
    *RW_LOCK_SOURCE.write() = source;
}

// Private

const MILLISECONDS_PER_SECOND: u64 = 1_000;
const MICROSECONDS_PER_MILLISECOND: u64 = 1_000;
const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;
const NANOSECONDS_PER_MILLISECONDS: u64 =
    NANOSECONDS_PER_MICROSECOND * MICROSECONDS_PER_MILLISECOND;

#[cfg(not(target_arch = "wasm32"))]
pub fn default_source() -> Milliseconds {
    START.elapsed().as_millis() as Milliseconds
}

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
lazy_static! {
    static ref START: Instant = Instant::now();
}

#[cfg(target_arch = "wasm32")]
pub fn default_source() -> Milliseconds {
    panic!("No default source for `wasm32`.  Call `lumen_runtime::time::monotonic::set_source(millisecond_source)`")
}

lazy_static! {
    static ref RW_LOCK_SOURCE: RwLock<Source> = RwLock::new(default_source);
}
