pub mod dynamic_call;

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(target_env = "wasi")] {
        mod wasi;
        use self::wasi as arch;
    } else {
        mod unknown;
        use self::unknown as arch;
    }
}

pub use self::arch::alloc;
pub use self::arch::sysconf;
