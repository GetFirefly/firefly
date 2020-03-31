use super::Milliseconds;
use js_sys::Date;

pub fn time_in_milliseconds() -> Milliseconds {
    Date::now() as Milliseconds
}
