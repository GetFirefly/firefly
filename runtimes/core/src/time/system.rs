use num_bigint::BigInt;

use liblumen_alloc::erts::time::Milliseconds;

use crate::time::{self, Unit};

pub fn time_in_unit(unit: Unit) -> BigInt {
    let system = time();
    time::convert_milliseconds(system.into(), unit)
}

#[cfg(not(all(target_arch = "wasm32", feature = "time_web_sys")))]
mod sys {
    use std::time::SystemTime;

    use super::*;

    pub fn time() -> System {
        System(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        )
    }
}

#[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))]
mod sys {
    use js_sys::Date;

    use super::*;

    pub fn time() -> System {
        System(Date::now() as u64)
    }
}

pub use self::sys::*;

pub struct System(u64);

impl From<System> for Milliseconds {
    fn from(system: System) -> Self {
        Self(system.0)
    }
}
