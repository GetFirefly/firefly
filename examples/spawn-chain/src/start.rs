#[cfg(target_arch = "wasm32")]
use lumen_runtime::time;
#[cfg(target_arch = "wasm32")]
use lumen_runtime::time::monotonic::Milliseconds;

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
pub fn set_time_monotonic_source() {
    time::monotonic::set_source(time_monotonic_source);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn set_time_monotonic_source() {
    // use the default source that works when not on wasm32
}

#[cfg(target_arch = "wasm32")]
pub fn log_1(string: String) {
    web_sys::console::log_1(&string.into());
}

#[cfg(not(target_arch = "wasm32"))]
pub fn log_1(string: String) {
    println!("{}", string);
}

#[cfg(target_arch = "wasm32")]
fn time_monotonic_source() -> Milliseconds {
    let window = web_sys::window().expect("should have a window in this context");
    let performance = window
        .performance()
        .expect("performance should be available");

    performance.now() as Milliseconds
}
