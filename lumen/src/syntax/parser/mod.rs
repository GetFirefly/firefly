//! Erlang source code parser
//!
//! # Examples
//!
//!
//!     use ::syntax::parser::{Parser, TokenReader};
//!     use ::syntax::parser::cst::Expr;
//!     use ::syntax::preprocessor::Preprocessor;
//!     use ::syntax::tokenizer::Lexer;
//!
//!     let text = r#"io:format("Hello World")"#;
//!     let mut parser = Parser::new(TokenReader::new(Preprocessor::new(Lexer::new(text))));
//!     parser.parse::<Expr>().unwrap();
//! ```
pub use self::error::{Error, ErrorKind};
pub use self::parser::Parser;
pub use self::token_reader::TokenReader;

pub mod builtin;
pub mod cst;
pub mod traits;

mod error;
mod parser;
mod token_reader;

#[cfg(test)]
mod test;

/// This crate specific `Result` type.
pub type Result<T> = ::std::result::Result<T, Error>;
