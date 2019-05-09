#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(alloc)]
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(stmt_expr_attributes)]
#![feature(exclusive_range_pattern)]
#![feature(ptr_offset_from)]

extern crate alloc;

#[macro_use]
pub mod utils;
mod blocks;
mod carriers;
mod mmap;
mod sorted;
mod sys;
mod sys_alloc;
//mod size_classes;
mod std_alloc;
//mod fixed_alloc;
mod erts;

/// The system allocator. Can be used with `#[global_allocator]`, like so:
///
/// ```ignore
/// #[global_allocator]
/// pub static ALLOC: SysAlloc = SysAlloc;
/// ```
pub use sys_alloc::SysAlloc;

/// The standard allocator. Used for general purpose allocations
pub use std_alloc::StandardAlloc;

// A fixed size allocator. Used for allocations that fall within predictable size classes.
//pub use fixed_alloc::FixedAlloc;

// Runtime system support, e.g. mutexes, process heaps, etc.
pub use erts::*;
