pub mod mmap;
mod region;
pub mod size_classes;
mod sys_alloc;
pub mod utils;

pub use self::region::Region;
pub use self::sys_alloc::*;

// Re-export core alloc types
pub mod prelude {
    pub use core::alloc::{AllocError, Allocator, GlobalAlloc, Layout, LayoutError};
    pub use core::ptr::NonNull;
    pub use core_alloc::alloc::{handle_alloc_error, Global};
}

pub use self::prelude::*;
