use std::time::SystemTime;

use super::Milliseconds;

pub fn time_in_milliseconds() -> Milliseconds {
    SystemTime::now().elapsed().unwrap().as_millis() as Milliseconds
}
