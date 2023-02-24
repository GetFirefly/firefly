#![deny(warnings)]
#![feature(iterator_try_collect)]
#![feature(box_patterns)]
#![feature(slice_take)]

mod ir;
pub mod macros;
pub mod passes;
pub mod printer;

pub use self::ir::*;
