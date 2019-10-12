use chrono::prelude::*;

pub fn get_utc_now() -> [usize; 6] {
    let utc: DateTime<Utc> = Utc::now();

    [
        utc.year() as usize,
        utc.month() as usize,
        utc.day() as usize,
        utc.hour() as usize,
        utc.minute() as usize,
        utc.second() as usize,
    ]
}
