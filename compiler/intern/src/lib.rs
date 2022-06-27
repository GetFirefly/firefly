#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(test)]

extern crate alloc;

#[cfg(any(test, bench))]
extern crate test;

pub mod arena;
pub mod symbol;
pub mod symbols;

pub use symbol::{Ident, InternedString, LocalInternedString, Symbol};
