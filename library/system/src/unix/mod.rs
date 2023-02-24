pub mod alloc;
pub mod mmap;

mod sysconf;

pub use self::sysconf::*;
pub use std::time;
