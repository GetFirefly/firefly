use chrono::prelude::*;
use std::convert::TryInto;

pub fn get_utc_now() -> ((i32, i32, i32), (i32, i32, i32)) {
    let utc: DateTime<Utc> = Utc::now();
    (
        (
            utc.year().try_into().unwrap(),
            utc.month().try_into().unwrap(),
            utc.day().try_into().unwrap(),
        ),
        (
            utc.hour().try_into().unwrap(),
            utc.minute().try_into().unwrap(),
            utc.second().try_into().unwrap(),
        ),
    )
}
