#[cfg_attr(target_env = "wasi", path = "wasi/mod.rs")]
#[cfg_attr(not(target_env = "wasi"), path = "unknown/mod.rs")]
mod os;

pub use self::os::*;
