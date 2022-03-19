#![deny(warnings)]
#![cfg_attr(not(test), no_std)]
#![feature(test)]
// Used for unlikely intrinsic
#![feature(core_intrinsics)]
// Used for allocators
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
// Dynamic dispatch intrinsics
#![feature(c_unwind)]
#![feature(naked_functions)]

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
