#![recursion_limit = "128"]
#![allow(stable_features)]
#![feature(core_intrinsics)]
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(ptr_offset_from)]
#![feature(exact_size_is_empty)]
#![feature(type_alias_enum_variants)]
#![feature(alloc)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc;

mod blocks;
mod borrow;
mod carriers;
mod erts;
mod stats_alloc;
mod segmented_alloc;
mod size_class_alloc;
mod sorted;
pub mod stats;
pub mod std_alloc;

/// The system allocator. Can be used with `#[global_allocator]`, like so:
///
/// ```ignore
/// #[global_allocator]
/// pub static ALLOC: SysAlloc = SysAlloc;
/// ```
pub use liblumen_core::alloc::SysAlloc;

/// A tracing allocator for tracking statistics about the allocator it wraps
pub use self::stats_alloc::StatsAlloc;

// An allocator that uses segmented sub-allocators to more efficiently manage
// allocations of variable sizes that fall within predictable size ranges
pub use self::segmented_alloc::SegmentedAlloc;

// An allocator that manages buckets of slab allocators as a highly efficient
// means of managing allocations with fixed sizes
pub use self::size_class_alloc::SizeClassAlloc;

// Runtime system support, e.g. process heaps, etc.
pub use erts::*;

/// Provides information about an allocator from `liblumen_alloc`
#[derive(Debug)]
pub struct AllocatorInfo {
    num_multi_block_carriers: usize,
    num_single_block_carriers: usize,
}
