#![no_std]
// Used for access to pointer metadata
#![feature(ptr_metadata)]
// Used for implementing Fn, etc. for GcBox, as
// well as interop with closures produced by the compiler
#![feature(fn_traits)]
#![feature(unboxed_closures)]
// Used for const TypeId::of::<T>()
#![feature(const_type_id)]

extern crate alloc;
#[cfg(test)]
extern crate test;

pub mod cmp;
pub mod error;
pub mod function;
pub mod process;
pub mod term;
