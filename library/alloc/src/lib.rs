#![no_std]
// Used for abort
#![feature(core_intrinsics)]
// Used for allocators
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
// Used for access to pointer metadata
#![feature(ptr_metadata)]
#![feature(pointer_byte_offsets)]
// Used for access to size/alignment information for raw pointers
#![feature(layout_for_ptr)]
#![feature(ptr_sub_ptr)]
// Used for the Unsize marker trait
#![feature(unsize)]
// For specializing the WriteCloneIntoRaw trait
#![feature(min_specialization)]
#![cfg_attr(test, feature(test))]

extern crate alloc;
#[cfg(test)]
extern crate std;
#[cfg(test)]
extern crate test;

pub mod allocators;
pub mod clone;
pub mod fragment;
pub mod heap;
pub mod mmap;
mod utils;

pub use self::utils::*;
