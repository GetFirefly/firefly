//! Unwinding for *wasm32* target.
//!
//! This requires the exception-handling feature to be enabled

use alloc::boxed::Box;

use super::cleanup as cleanup_panic;
use super::ErlangPanic;

#[repr(C)]
pub struct Exception {
    cause: *mut ErlangPanic,
}

extern "C" {
    #[link_name = "__builtin_wasm_throw"]
    fn wasm_throw(index: u32, exception: *mut u8) -> !;

    #[link_name = "__builtin_wasm_rethrow_in_catch"]
    fn wasm_rethrow() -> !;
}

#[inline]
pub unsafe fn panic(data: *mut ErlangPanic) -> u32 {
    let exception = Box::new(Exception { cause: data });
    wasm_throw(0, Box::into_raw(exception) as *mut _)
}

pub unsafe fn cause(ptr: *mut u8) -> *mut ErlangPanic {
    let exception = &*(ptr as *mut ErlangPanic);
    exception.cause
}

#[inline]
pub unsafe fn cleanup(ptr: *mut u8) {
    let exception = Box::from_raw(ptr as *mut Exception);
    cleanup_panic(exception.cause);
}

#[no_mangle]
pub unsafe extern "C-unwind" fn firefly_eh_unwind_resume(_panic_ctx: *mut u8) -> ! {
    wasm_rethrow()
}
