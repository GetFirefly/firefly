#![feature(test)]
#![feature(alloc)]
#![feature(dropck_eyepatch)]
#![feature(core_intrinsics)]
#![feature(raw_vec_internals)]
#![feature(const_fn)]
#![feature(optin_builtin_traits)]
#![feature(nll)]

extern crate alloc;
#[cfg(test)] extern crate test;

mod arena;

pub use self::arena::*;
