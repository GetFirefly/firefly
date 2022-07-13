#![no_std]
#![feature(link_cfg)]
#![feature(nll)]
#![feature(c_unwind)]
#![link(kind = "static", modifiers = "-bundle")]
#![cfg_attr(not(target_env = "msvc"), feature(libc))]

cfg_if::cfg_if! {
    if #[cfg(target_env = "msvc")] {
        // no extra unwinder support needed
    } else if #[cfg(any(
        target_os = "l4re",
        target_os = "none",
    ))] {
        // These "unix" family members do not have unwinder.
        // Note this also matches x86_64-linux-kernel.

    } else if #[cfg(any(
        unix,
        windows,
        target_os = "cloudabi",
        all(target_vendor = "fortanix", target_env = "sgx"),
    ))] {
        mod libunwind;
        pub use libunwind::*;
    } else {
        // no unwinder on the system!
        // - wasm32 (not emscripten, which is "unix" family)
        // - os=none ("bare metal" targets)
        // - os=hermit
        // - os=uefi
        // - os=cuda
        // - nvptx64-nvidia-cuda
        // - mipsel-sony-psp
        // - Any new targets not listed above.
    }
}

#[cfg(target_env = "musl")]
#[link(name = "unwind", kind = "static", cfg(target_feature = "crt-static"))]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(target_os = "redox")]
#[link(
    name = "gcc_eh",
    kind = "static-nobundle",
    cfg(target_feature = "crt-static")
)]
#[link(name = "gcc_s", cfg(not(target_feature = "crt-static")))]
extern "C" {}

#[cfg(all(target_vendor = "fortanix", target_env = "sgx"))]
#[link(name = "unwind", kind = "static-nobundle")]
extern "C" {}
