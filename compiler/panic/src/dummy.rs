//! Unwinding for *wasm32* target.
//!
//! Right now we don't support this, so this is just stubs.

use alloc::boxed::Box;
use core::any::Any;
use core::intrinsics;

pub unsafe fn cleanup(_ptr: *mut u8) -> usize {
    intrinsics::abort()
}

pub unsafe fn panic(_data: usize) -> u32 {
    intrinsics::abort()
}
