use num_bigint::BigInt;

use crate::time::{convert_milliseconds, Unit};
use liblumen_alloc::erts::time::Monotonic;

cfg_if::cfg_if! {
  if #[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))] {
     mod web_sys;
     pub use self::web_sys::*;
  } else {
     mod std;
     pub use self::std::*;
  }
}

pub fn time_in_unit(unit: Unit) -> BigInt {
    let monotonic = time();
    let milliseconds = monotonic.into();
    convert_milliseconds(milliseconds, unit)
}
