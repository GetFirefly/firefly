#![no_std]
#![feature(core_intrinsics)]
#![feature(test)]
#![feature(once_cell)]
// Used to provide low-level alloc primitives
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
// Used for OnceLock impl
#![feature(const_default_impls)]
#![feature(const_trait_impl)]
#![feature(dropck_eyepatch)]
#![feature(never_type)]

/// When possible, we also link against libstd to take advantage of functionality provided there
#[cfg(any(unix, windows, target_os = "wasi", target_family = "wasm"))]
extern crate std;

#[cfg(test)]
extern crate test;

/// The minimum alignment we use on all platforms
pub const MIN_ALIGN: usize = 8;

cfg_if::cfg_if! {
    if #[cfg(unix)] {
        #[path = "unix/mod.rs"]
        pub mod arch;
    } else if #[cfg(windows)] {
        #[path = "windows/mod.rs"]
        pub mod windows;
    } else if #[cfg(target_os = "wasi")] {
        #[path = "wasi/mod.rs"]
        pub mod wasi;
    } else if #[cfg(target_family = "wasm")] {
        #[path = "wasm/mod.rs"]
        pub mod wasm;
    } else {
        compile_error!("unsupported target platform");
    }
}

pub mod alloc;
pub mod mem;
pub mod sync;
pub mod time;
