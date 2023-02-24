#![feature(dropck_eyepatch)]
#![cfg_attr(any(test, bench), feature(test))]
#![no_std]

extern crate alloc;

#[cfg(any(test, bench))]
extern crate test;

pub mod arena;
pub mod symbol;
pub mod symbols;

pub use symbol::{Ident, InternedString, LocalInternedString, Symbol};
