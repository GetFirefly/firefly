use std::time::SystemTime;

use super::Milliseconds;

pub fn time_in_milliseconds() -> Milliseconds {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as Milliseconds
}
