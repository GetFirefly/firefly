#![allow(dead_code)]
use lazy_static::lazy_static;
use std::time::{Duration, Instant, SystemTime};

lazy_static! {
    static ref SYSTEM_START_TIME: Instant = { Instant::now() };
}

pub struct ErlangTimestamp {
    pub megaseconds: u32,
    pub seconds: u32,
    pub microseconds: u32,
}

impl ErlangTimestamp {
    pub fn from_duration(duration: Duration) -> Self {
        let microseconds = duration.subsec_micros();
        let total_secs = duration.as_secs();
        let megaseconds = (total_secs / 1_000_000) as u32;
        let seconds = (total_secs % 1_000_000) as u32;
        Self {
            megaseconds,
            seconds,
            microseconds,
        }
    }
}

#[inline]
pub fn system_time() -> Duration {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("Unable to get system time!")
}

#[inline]
pub fn monotonic_time() -> Duration {
    Instant::now().duration_since(*SYSTEM_START_TIME)
}
