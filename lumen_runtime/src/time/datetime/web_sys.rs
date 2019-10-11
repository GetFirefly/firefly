pub fn get_utc_now() -> ((i32, i32, i32), (i32, i32, i32)) {
    let now = js_sys::Date::new_0();

    (
        (
            now.get_utc_full_year(),
            now.get_utc_month(),
            now.get_utc_date(),
        ),
        (
            now.get_utc_hours(),
            now.get_utc_minutes(),
            now.get_utc_seconds(),
        ),
    )
}
