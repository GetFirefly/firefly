use crate::code;

pub fn set_apply_fn() {
    lumen_runtime::code::set_apply_fn(code::apply)
}

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
pub fn set_parking_lot_time_now_fn() {
    parking_lot_core::time::set_now_fn(now);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn set_parking_lot_time_now_fn() {
    // use the default that works when not on wasm32
}

#[cfg(target_arch = "wasm32")]
fn now() -> parking_lot_core::time::Instant {
    let window = web_sys::window().expect("should have a window in this context");
    let performance = window
        .performance()
        .expect("performance should be available");

    parking_lot_core::time::Instant::from_millis(performance.now() as u64)
}
