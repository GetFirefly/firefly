//! Unwinding for targets without unwind support
//!
//! Since we can't raise exceptions on these platforms, they simply abort
use core::intrinsics;

use super::ErlangPanic;

pub unsafe fn cause(_ptr: *mut u8) -> *mut ErlangPanic {
    intrinsics::abort()
}

pub unsafe fn cleanup(_ptr: *mut u8) {
    intrinsics::abort()
}

pub unsafe fn panic(_data: *mut ErlangPanic) -> u32 {
    intrinsics::abort()
}
