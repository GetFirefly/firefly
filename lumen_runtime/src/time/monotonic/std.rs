use std::time::Instant;

use super::Milliseconds;

pub fn time_in_milliseconds() -> Milliseconds {
    START.elapsed().as_millis() as Milliseconds
}

lazy_static! {
    static ref START: Instant = Instant::now();
}
