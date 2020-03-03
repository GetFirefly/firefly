#![cfg_attr(not(test), no_std)]
#![feature(test)]
#![feature(core_intrinsics)]
// Used for allocators
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
// The following are used for our re-implementations
// of alloc types parameterized on their backing allocator
#![feature(specialization)]
#![feature(dropck_eyepatch)]
#![feature(trusted_len)]
#![feature(exact_size_is_empty)]
#![feature(try_reserve)]
#![feature(ptr_internals)]
#![feature(ptr_offset_from)]
#![feature(slice_partition_dedup)]
// Dynamic dispatch intrinsics
#![feature(asm)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc as core_alloc;

#[cfg(test)]
extern crate test;

pub mod alloc;
pub mod atoms;
pub mod cmp;
pub mod locks;
pub mod symbols;
pub mod sys;
pub mod util;

pub use liblumen_core_macros::entry;
