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
#![feature(unwind_attributes)]
#![feature(abi_thiscall)]
#![feature(rustc_attrs)]
#![feature(raw)]
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

// Entry point for raising an exception, just delegates to the platform-specific implementation.
#[unwind(allowed)]
#[no_mangle]
pub unsafe extern "C" fn __lumen_start_panic(payload: usize) -> u32 {
    imp::panic(payload)
}

#[no_mangle]
pub unsafe extern "C" fn __lumen_get_exception(ptr: *mut u8) -> usize {
    imp::cleanup(ptr)
}
