#![deny(warnings)]
#![feature(iterator_try_collect)]
#![feature(map_try_insert)]
#![feature(box_patterns)]
#![feature(assert_matches)]

mod bimap;
mod ir;
pub mod macros;
pub mod passes;
mod printer;

pub use self::bimap::{BiMap, Name};
pub use self::ir::*;
