// Do not fail the build when feature flags are stabilized on recent nightlies, just warn
#![allow(stable_features)]
// Support backtraces in errors
#![feature(backtrace)]
// Allow use of intrinsics, e.g. unlikely/copy_nonoverlapping/etc.
#![feature(core_intrinsics)]
// Allocator APIs
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
// Support offset_from pointer calculation
#![feature(ptr_offset_from)]
// Support specialization of traits
#![feature(specialization)]
// Support SliceIndex trait
#![feature(slice_index_methods)]
#![feature(trait_alias)]
#![feature(raw_vec_internals)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc;

#[cfg(target_arch = "wasm32")]
extern crate wasm_bindgen_test;

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
pub use self::size_class_alloc::{SizeClassAlloc, SizeClassAllocRef};

// Runtime system support, e.g. process heaps, etc.
pub use erts::*;

pub use borrow::CloneToProcess;

/// Provides information about an allocator from `liblumen_alloc`
#[derive(Debug)]
pub struct AllocatorInfo {
    num_multi_block_carriers: usize,
    num_single_block_carriers: usize,
}
