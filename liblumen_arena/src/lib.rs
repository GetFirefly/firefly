#![no_std]
#![feature(test)]
// Used for #[may_dangle] on TypedArena
#![feature(dropck_eyepatch)]
// Used for arith_offset
#![feature(core_intrinsics)]
#![feature(new_uninit)]
#![feature(maybe_uninit_slice)]
// Used for the implementation of the arenas
#![feature(raw_vec_internals)]

extern crate alloc;
#[cfg(test)]
extern crate test;

mod arena;

pub use self::arena::*;
