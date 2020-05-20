#![cfg_attr(not(test), no_std)]
#![feature(test)]
#![feature(core_intrinsics)]
// Used for allocators
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(ptr_offset_from)]
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
