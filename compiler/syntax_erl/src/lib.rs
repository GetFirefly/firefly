#![deny(warnings)]
#![feature(trait_alias)]
#![feature(generic_associated_types)]
#![feature(map_first_last)]
#![feature(map_try_insert)]
#![feature(let_else)]
#![feature(box_patterns)]

#[macro_use]
mod macros;
mod ast;
mod evaluator;
mod lexer;
mod parser;
pub mod passes;
mod preprocessor;
mod util;
mod visit;

pub use self::ast::*;
pub use self::lexer::*;
pub use self::parser::*;
pub use self::preprocessor::*;
