//#![deny(warnings)]
#![feature(trait_alias)]
#![feature(never_type)]
#![feature(generic_associated_types)]
#![feature(iterator_try_collect)]
#![feature(map_first_last)]
#![feature(map_try_insert)]
#![feature(drain_filter)]
#![feature(let_else)]
#![feature(box_patterns)]
#![feature(slice_take)]
#![feature(assert_matches)]

#[macro_use]
mod macros;
pub mod ast;
pub mod cst;
mod evaluator;
pub mod kernel;
mod lexer;
mod parser;
pub mod passes;
mod preprocessor;
mod util;
pub mod visit;

pub use self::ast::{Arity, Name};
pub use self::lexer::*;
pub use self::parser::*;
pub use self::preprocessor::*;
