pub fn get_utc_now() -> [usize; 6] {
    let now = js_sys::Date::new_0();

    return [
        now.get_utc_full_year() as usize,
        now.get_utc_month() as usize,
        now.get_utc_date() as usize,
        now.get_utc_hours() as usize,
        now.get_utc_minutes() as usize,
        now.get_utc_seconds() as usize,
    ];
}
