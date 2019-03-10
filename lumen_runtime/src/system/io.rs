#[cfg(not(target_arch = "wasm32"))]
use std::ffi::CStr;

#[cfg(not(target_arch = "wasm32"))]
use libc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    pub fn console_log(s: &str);
}

#[allow(dead_code)]
#[no_mangle]
// libc::c_char does not exist for `wasm32-unknown-unknown`
#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn lumen_system_io_puts(s: *const libc::c_char) {
    let sref = unsafe { CStr::from_ptr(s).to_string_lossy() };
    puts(&sref);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn puts(s: &str) {
    println!("{}", s);
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub fn puts(s: &str) {
    console_log(s);
}
