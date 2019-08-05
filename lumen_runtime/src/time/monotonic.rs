use num_bigint::BigInt;

use crate::time::convert;
use crate::time::Unit::{self, *};

cfg_if::cfg_if! {
  if #[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))] {
     mod web_sys;
     pub use self::web_sys::*;
  } else {
     mod std;
     pub use self::std::*;
  }
}

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

// Private

const MILLISECONDS_PER_SECOND: u64 = 1_000;
const MICROSECONDS_PER_MILLISECOND: u64 = 1_000;
const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;
const NANOSECONDS_PER_MILLISECONDS: u64 =
    NANOSECONDS_PER_MICROSECOND * MICROSECONDS_PER_MILLISECOND;
