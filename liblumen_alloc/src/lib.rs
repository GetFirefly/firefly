#![recursion_limit = "128"]
//#![cfg_attr(not(test), no_std)]
// Do not fail the build when feature flags are stabilized on recent nightlies, just warn
#![allow(stable_features)]
// Allow use of intrinsics, e.g. unlikely/copy_nonoverlapping/etc.
#![feature(core_intrinsics)]
// Allocator APIs
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
// Support offset_from pointer calculation
#![feature(ptr_offset_from)]
// Support is_empty for ExactSizeIterator
#![feature(exact_size_is_empty)]
// Support use of Self and other type aliases in matches on enum variants
#![feature(type_alias_enum_variants)]
// For static assertions that use logical operators
#![feature(const_fn)]
// Allow `[#cfg(debug_assertions)]` to enable file, line, and column for runtime::Exception
#![feature(param_attrs)]
#![feature(underscore_const_names)]
#![feature(const_compare_raw_pointers)]
#![feature(const_raw_ptr_to_usize_cast)]
#![feature(const_raw_ptr_deref)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc;

#[macro_use]
extern crate static_assertions;

#[macro_use]
mod macros;

mod blocks;
pub mod borrow;
mod carriers;
pub mod erts;
mod mem;
mod segmented_alloc;
mod size_class_alloc;
mod sorted;
pub mod stats;
mod stats_alloc;
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

pub use borrow::CloneToProcess;

/// Provides information about an allocator from `liblumen_alloc`
#[derive(Debug)]
pub struct AllocatorInfo {
    num_multi_block_carriers: usize,
    num_single_block_carriers: usize,
}
