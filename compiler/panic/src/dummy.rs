//! Unwinding for targets without unwind support
//!
//! Since we can't raise exceptions on these platforms, they simply abort
use core::intrinsics;

pub unsafe fn cleanup(_ptr: *mut u8) -> usize {
    intrinsics::abort()
}

pub unsafe fn panic(_data: usize) -> u32 {
    intrinsics::abort()
}
