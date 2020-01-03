#![allow(dead_code)]
use num_bigint::{BigInt, ToBigInt};
use num_traits::cast::ToPrimitive;
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

    pub fn from_microseconds(system_time: &BigInt) -> Self {
        // algorithm taken from http://erlang.org/doc/man/erlang.html#timestamp-0
        let megaseconds: BigInt = system_time / ((1000000000000 as u64).to_bigint().unwrap());
        let seconds: BigInt = system_time / 1000000 - &megaseconds * 1000000;
        let microseconds: BigInt = system_time % 1000000;

        Self {
            megaseconds: megaseconds.to_u32().unwrap(),
            seconds: seconds.to_u32().unwrap(),
            microseconds: microseconds.to_u32().unwrap(),
        }
    }
}

#[inline]
pub fn system_time() -> Duration {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("Unable to get system time!")
}
