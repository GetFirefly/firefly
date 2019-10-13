use chrono::prelude::*;

pub fn get_local_date() -> [usize; 3] {
    let datetime: [usize; 6] = get_local_now();
    [datetime[0], datetime[1], datetime[2]]
}

pub fn get_local_time() -> [usize; 3] {
    let datetime: [usize; 6] = get_local_now();
    [datetime[3], datetime[4], datetime[5]]
}

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
