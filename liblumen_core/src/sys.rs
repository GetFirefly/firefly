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

pub mod dynamic_call {
    pub use super::arch::dynamic_call::*;

    pub type DynamicCallee = extern "C" fn() -> usize;
}

pub mod sysconf {
    use lazy_static::lazy_static;

    use super::arch::sysconf;

    /// The minimum alignment required for all targets
    pub const MIN_ALIGN: usize = 8;

    lazy_static! {
        static ref PAGE_SIZE: usize = sysconf::pagesize();
    }

    lazy_static! {
        static ref NUM_CPUS: usize = sysconf::get_num_cpus();
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
