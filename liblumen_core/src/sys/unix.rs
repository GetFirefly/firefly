pub mod alloc;
#[cfg(has_mmap)]
pub mod mmap;
pub mod sysconf;
#[cfg(target_arch = "x86_64")]
pub mod dynamic_call;
