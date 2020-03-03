pub mod alloc;
#[cfg(target_arch = "x86_64")]
pub mod dynamic_call;
#[cfg(has_mmap)]
pub mod mmap;
pub mod sysconf;
