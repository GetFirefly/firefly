use cfg_if::cfg_if;

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

pub use self::arch::alloc;
#[cfg(has_mmap)]
pub use self::arch::mmap;

pub mod sysconf {
    use cfg_if::cfg_if;
    use lazy_static::lazy_static;

    use super::arch::sysconf;

    // The minimum alignment guaranteed by the target
    cfg_if! {
        if #[cfg(all(any(
                target_arch = "x86",
                target_arch = "arm",
                target_arch = "mips",
                target_arch = "powerpc",
                target_arch = "powerpc64",
                target_arch = "asmjs",
                target_arch = "wasm32"
            )))] {
            pub const MIN_ALIGN: usize = 8;
            pub const UNUSED_PTR_BITS: usize = 3;
        } else if #[cfg(all(any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "mips64",
                target_arch = "s390x",
                target_arch = "sparc64"
            )))] {
            pub const MIN_ALIGN: usize = 16;
            pub const UNUSED_PTR_BITS: usize = 4;
        }
    }

    lazy_static! {
        static ref PAGE_SIZE: usize = { sysconf::pagesize() };
    }

    lazy_static! {
        static ref NUM_CPUS: usize = { sysconf::get_num_cpus() };
    }

    /// Returns the current page size in bytes
    #[inline(always)]
    pub fn pagesize() -> usize {
        *PAGE_SIZE
    }

    #[inline(always)]
    pub fn num_cpus() -> usize {
        *NUM_CPUS
    }
}
