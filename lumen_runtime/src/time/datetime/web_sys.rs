use js_sys::Date;

pub fn get_local_now() -> [usize; 6] {
    let now = Date::new_0();

    [
        now.get_full_year() as usize,
        (now.get_month() as usize) + 1, // Since months in javascript are 0-based
        now.get_date() as usize,
        now.get_hours() as usize,
        now.get_minutes() as usize,
        now.get_seconds() as usize,
    ]
}

pub fn get_utc_now() -> [usize; 6] {
    let now = Date::new_0();

    [
        now.get_utc_full_year() as usize,
        (now.get_utc_month() as usize) + 1, // Since months in javascript are 0-based
        now.get_utc_date() as usize,
        now.get_utc_hours() as usize,
        now.get_utc_minutes() as usize,
        now.get_utc_seconds() as usize,
    ]
}
