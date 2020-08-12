use super::Monotonic;

pub fn time() -> Monotonic {
    let window = web_sys::window().expect("should have a window in this context");
    let performance = window
        .performance()
        .expect("performance should be available");

    Monotonic(performance.now() as u64)
}
