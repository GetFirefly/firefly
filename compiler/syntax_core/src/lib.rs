#![deny(warnings)]
#![feature(generic_associated_types)]
#![feature(iterator_try_collect)]
#![feature(map_first_last)]
#![feature(let_else)]
#![feature(box_patterns)]
#![feature(slice_take)]

mod ir;
pub mod macros;
pub mod passes;
pub mod printer;

pub use self::ir::*;
