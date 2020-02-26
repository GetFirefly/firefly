use num_bigint::BigInt;

use lazy_static::lazy_static;

use crate::time::{self, Milliseconds, Unit};

pub fn time(unit: Unit) -> BigInt {
    let milliseconds = time_in_milliseconds();
    time::convert_milliseconds(milliseconds, unit)
}

#[cfg(not(all(target_arch = "wasm32", feature = "time_web_sys")))]
mod sys {
    use std::time::Instant;

    use super::*;

    pub fn time_in_milliseconds() -> Milliseconds {
        START.elapsed().as_millis() as Milliseconds
    }

    lazy_static! {
        static ref START: Instant = Instant::now();
    }
}

#[cfg(all(target_arch = "wasm32", feature = "time_web_sys"))]
mod sys {
    use super::*;

    pub fn time_in_milliseconds() -> Milliseconds {
        let window = web_sys::window().expect("should have a window in this context");
        let performance = window
            .performance()
            .expect("performance should be available");

        performance.now() as Milliseconds
    }
}

pub use self::sys::*;
