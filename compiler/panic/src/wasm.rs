//! Unwinding for *wasm32* target.
//!
//! This requires the exception-handling feature to be enabled

use alloc::boxed::Box;

#[repr(C)]
pub struct Exception {
    cause: usize,
}

extern "C" {
    #[link_name = "__builtin_wasm_throw"]
    fn wasm_throw(index: u32, exception: *mut u8) -> !;

    #[link_name = "__builtin_wasm_rethrow_in_catch"]
    fn wasm_rethrow() -> !;
}

#[inline]
pub unsafe fn panic(data: usize) -> u32 {
    let exception = Box::new(Exception {
        cause: data,
    });
    wasm_throw(0, Box::into_raw(exception) as *mut _)
}

#[inline]
pub unsafe fn cleanup(_ptr: *mut u8) -> usize {
    let exception = Box::from_raw(ptr as *mut Exception);
    exception.cause
}

#[no_mangle]
#[unwind(allowed)]
pub unsafe extern "C" fn lumen_eh_unwind_resume(_panic_ctx: *mut u8) -> ! {
    wasm_rethrow()
}

