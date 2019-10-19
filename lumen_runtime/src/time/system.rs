use num_bigint::BigInt;

use crate::time::convert;
use crate::time::Milliseconds;
use crate::time::Unit::{self, *};
use crate::time::{
    MICROSECONDS_PER_MILLISECOND, MILLISECONDS_PER_SECOND, NANOSECONDS_PER_MILLISECONDS,
};

cfg_if::cfg_if! {
  if #[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))] {
     mod web_sys;
     pub use self::web_sys::*;
  } else {
     mod std;
     pub use self::std::*;
  }
}

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
