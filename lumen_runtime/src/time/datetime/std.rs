use chrono::prelude::*;

pub fn get_local_now() -> [usize; 6] {
    datetime_to_array(Local::now())
}

pub fn get_utc_now() -> [usize; 6] {
    datetime_to_array(Utc::now())
}

fn datetime_to_array<Tz: TimeZone>(datetime: DateTime<Tz>) -> [usize; 6] {
    [
        datetime.year() as usize,
        datetime.month() as usize,
        datetime.day() as usize,
        datetime.hour() as usize,
        datetime.minute() as usize,
        datetime.second() as usize,
    ]
}
