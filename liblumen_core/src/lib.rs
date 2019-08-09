#![cfg_attr(not(test), no_std)]
#![feature(core_intrinsics)]
#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(specialization)]
#![feature(dropck_eyepatch)]
#![feature(trusted_len)]
#![feature(raw_vec_internals)]
#![feature(exact_size_is_empty)]
#![feature(try_reserve)]
#![feature(ptr_internals)]
#![feature(ptr_offset_from)]
#![feature(slice_partition_dedup)]

#[cfg_attr(not(test), macro_use)]
extern crate alloc as core_alloc;

pub mod alloc;
pub mod locks;
pub mod sys;
pub mod util;
