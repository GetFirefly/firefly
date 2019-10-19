use num_bigint::BigInt;

use crate::time::{convert_milliseconds, Milliseconds, Unit};

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
    convert_milliseconds(milliseconds, unit)
}
