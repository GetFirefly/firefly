use std::time::Instant;

use num_bigint::BigInt;

use crate::time::convert;
use crate::time::Unit::{self, *};

// Must be at least a `u64` because `u32` is only ~49 days (`(1 << 32)`)
pub type Milliseconds = u64;

pub fn time(unit: Unit) -> BigInt {
    let duration = START.elapsed();

    match unit {
        Second => duration.as_secs().into(),
        Millisecond => duration.as_millis().into(),
        Microsecond => duration.as_micros().into(),
        Nanosecond => duration.as_nanos().into(),
        _ => convert(duration.as_nanos().into(), Nanosecond, unit),
    }
}

pub fn time_in_milliseconds() -> Milliseconds {
    START.elapsed().as_millis() as Milliseconds
}

lazy_static! {
    static ref START: Instant = Instant::now();
}
