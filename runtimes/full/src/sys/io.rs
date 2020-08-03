#[cfg(not(target_arch = "wasm32"))]
use std::ffi::CStr;

#[cfg(not(target_arch = "wasm32"))]
use libc;

pub use lumen_rt_core::sys::io::puts;

#[allow(dead_code)]
#[no_mangle]
// libc::c_char does not exist for `wasm32-unknown-unknown`
#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn lumen_system_io_puts(s: *const libc::c_char) {
    let sref = unsafe { CStr::from_ptr(s).to_string_lossy() };
    puts(&sref);
}
