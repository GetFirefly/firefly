pub mod alloc_ref;
pub mod boxed;
pub mod mmap;
pub mod raw_vec;
mod static_alloc;
mod sys_alloc;
pub mod utils;
pub mod vec;

pub use self::static_alloc::StaticAlloc;
pub use self::sys_alloc::*;
pub use self::utils as alloc_utils;
