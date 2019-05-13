#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(alloc)]
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(ptr_offset_from)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc;

mod blocks;
mod carriers;
mod sorted;
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
pub use liblumen_core::alloc::SysAlloc;

/// The standard allocator. Used for general purpose allocations
pub use std_alloc::StandardAlloc;

// A fixed size allocator. Used for allocations that fall within predictable size classes.
//pub use fixed_alloc::FixedAlloc;

// Runtime system support, e.g. process heaps, etc.
pub use erts::*;

/// Provides information about an allocator from `liblumen_alloc`
#[derive(Debug)]
pub struct AllocatorInfo {
    num_multi_block_carriers: usize,
    num_single_block_carriers: usize,
}
