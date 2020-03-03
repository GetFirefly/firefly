use num_bigint::BigInt;

use crate::time::{self, Milliseconds, Unit};

pub fn time(unit: Unit) -> BigInt {
    let milliseconds = time_in_milliseconds();
    time::convert_milliseconds(milliseconds, unit)
}

#[cfg(not(all(target_arch = "wasm32", feature = "time_web_sys")))]
mod sys {
    use std::time::SystemTime;

    use super::*;

    pub fn time_in_milliseconds() -> Milliseconds {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as Milliseconds
    }
}

#[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))]
mod sys {
    use js_sys::Date;

    use super::*;

    pub fn time_in_milliseconds() -> Milliseconds {
        Date::now() as Milliseconds
    }
}

pub use self::sys::*;
