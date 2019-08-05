use super::Milliseconds;

pub fn time_in_milliseconds() -> Milliseconds {
    let window = web_sys::window().expect("should have a window in this context");
    let performance = window
        .performance()
        .expect("performance should be available");

    performance.now() as Milliseconds
}
