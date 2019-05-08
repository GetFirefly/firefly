use cfg_if::cfg_if;
use lazy_static::lazy_static;

cfg_if! {
    if #[cfg(windows)] {
        mod windows;
        use self::windows as arch;
    } else if #[cfg(unix)] {
        mod unix;
        use self::unix as arch;
    } else if #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))] {
        mod wasm32;
        use self::wasm32 as arch;
    } else {
        compile_error!("unsupported platform!");
    }
}

pub(crate) use self::arch::*;

// The minimum alignment guaranteed by the architecture. This value is used to
// add fast paths for low alignment values.
#[cfg(all(any(
    target_arch = "x86",
    target_arch = "arm",
    target_arch = "mips",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "asmjs",
    target_arch = "wasm32"
)))]
pub const MIN_ALIGN: usize = 8;
#[cfg(all(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "mips64",
    target_arch = "s390x",
    target_arch = "sparc64"
)))]
pub const MIN_ALIGN: usize = 16;

lazy_static! {
    static ref PAGE_SIZE: usize = { self::arch::pagesize() };
}

lazy_static! {
    static ref PAGE_SIZE_MASK: usize = { *PAGE_SIZE - 1 };
}

/// Returns the current page size in bytes
#[allow(unused)]
#[inline(always)]
pub fn pagesize() -> usize {
    *PAGE_SIZE
}
