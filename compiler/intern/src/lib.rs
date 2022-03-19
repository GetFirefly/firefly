#![feature(core_intrinsics)]
#![feature(dropck_eyepatch)]
#![feature(test)]

extern crate alloc;

#[cfg(any(test, bench))]
extern crate test;

pub mod arena;

pub mod symbol;
pub use symbol::{symbols, Ident, InternedString, LocalInternedString, Symbol};
