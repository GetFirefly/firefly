#![deny(warnings)]
#![feature(generic_associated_types)]
#![feature(iterator_try_collect)]
#![feature(map_first_last)]
#![feature(map_try_insert)]
#![feature(let_else)]
#![feature(box_patterns)]
#![feature(assert_matches)]

mod bimap;
mod ir;
pub mod macros;
pub mod passes;
mod printer;

pub use self::bimap::{BiMap, Name};
pub use self::ir::*;
