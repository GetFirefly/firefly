use std::time::Instant;

use num_bigint::BigInt;

use crate::time::convert;
use crate::time::Unit::{self, *};

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

lazy_static! {
    static ref START: Instant = Instant::now();
}
