#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(alloc)]
#![feature(allocator_api)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc as core_alloc;

pub mod alloc;
pub mod locks;
pub mod sys;
