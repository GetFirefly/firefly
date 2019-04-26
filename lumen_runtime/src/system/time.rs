#![allow(dead_code)]
use std::time::{Duration, SystemTime};

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
