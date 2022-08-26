//! This crate supports exception handling in Erlang via panics/stack unwinding.
//!
//! The implementation makes use of whatever the native stack unwinding mechanism of
//! the target platform is.
//!
//! 1. MSVC targets use SEH in the `seh.rs` file.
//! 2. Emscripten uses C++ exceptions in the `emcc.rs` file.
//! 3. All other targets use libunwind/libgcc in the `gcc.rs` file.
//!
//! More documentation about each implementation can be found in the respective
//! module.

#![no_std]
#![feature(core_intrinsics)]
#![feature(lang_items)]
#![feature(libc)]
#![feature(c_unwind)]
#![feature(abi_thiscall)]
#![feature(rustc_attrs)]
#![panic_runtime]
#![feature(panic_runtime)]

extern crate alloc;

cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        #[path = "wasm.rs"]
        mod imp;
    } else if #[cfg(target_os = "hermit")] {
        #[path = "dummy.rs"]
        mod imp;
    } else if #[cfg(target_env = "msvc")] {
        #[path = "seh.rs"]
        mod imp;
    } else {
        // Rust runtime's startup objects depend on these symbols, so make them public.
        #[cfg(all(target_os="windows", target_arch = "x86", target_env="gnu"))]
        pub use real_imp::eh_frame_registry::*;
        #[path = "gcc.rs"]
        mod imp;
    }
}

mod dwarf;

/// This matches the structure of ErlangException in firefly_rt
#[repr(C)]
pub struct ErlangPanic {
    _header: usize,
    kind: usize,
    reason: usize,
    trace: *mut u8,
    _fragment: Option<*mut u8>,
}

extern "C-unwind" {
    // See `ErlangException` in firefly_rt
    #[allow(improper_ctypes)]
    #[link_name = "__firefly_cleanup_exception"]
    fn cleanup(ptr: *mut ErlangPanic);
}

// Entry point for raising an exception, just delegates to the platform-specific implementation.
#[no_mangle]
pub unsafe extern "C-unwind" fn __firefly_start_panic(payload: *mut ErlangPanic) -> u32 {
    imp::panic(payload)
}

#[no_mangle]
pub unsafe extern "C" fn __firefly_get_exception(ptr: *mut u8) -> *mut ErlangPanic {
    imp::cause(ptr)
}
