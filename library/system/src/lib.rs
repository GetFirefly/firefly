#![no_std]
#![feature(core_intrinsics)]
#![feature(test)]
#![feature(once_cell)]
// Used to provide low-level alloc primitives
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]

#[cfg(test)]
extern crate test;

pub mod arch {
    // Allow referencing each platform directly when conditionally compiling

    #[cfg(unix)]
    pub mod unix {
        pub use crate::unix::*;
    }

    #[cfg(all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        not(target_os = "emscripten")
    ))]
    pub mod wasm {
        pub use crate::wasm32::*;
    }

    #[cfg(windows)]
    pub mod windows {
        pub use crate::windows::*;
    }

    /// The minimum alignment on all platforms
    pub const MIN_ALIGN: usize = 8;

    // Re-export the current target platform under the `arch` namespace so that
    // shared functionality can be accessed without needing conditional compilation

    #[cfg(unix)]
    pub use self::unix::*;

    #[cfg(all(
        any(target_arch = "wasm32", target_arch = "wasm64"),
        not(target_os = "emscripten")
    ))]
    pub use self::wasm::*;

    #[cfg(windows)]
    pub use self::windows::*;
}

pub mod alloc;
pub mod cell;
pub mod mem;
pub mod sync;

#[cfg(unix)]
mod unix;

#[cfg(all(
    any(target_arch = "wasm32", target_arch = "wasm64"),
    not(target_os = "emscripten")
))]
mod wasm;

#[cfg(windows)]
mod windows;
