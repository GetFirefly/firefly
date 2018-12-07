#![allow(dead_code)]
use std::time::{Duration, Instant};

#[inline]
pub fn time<T>(fun: fn() -> T) -> (Duration, T) {
    let start = Instant::now();
    let result = fun();
    (start.elapsed(), result)
}
