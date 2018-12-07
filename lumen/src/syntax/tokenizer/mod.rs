//! Erlang source code tokenizer
//!
//! # Examples
//!
//! Tokenizes the Erlang code `io:format("Hello").`:
//!
//! ```
//! use erl_tokenize::Tokenizer;
//!
//! let src = r#"io:format("Hello")."#;
//! let tokenizer = Tokenizer::new(src);
//! let tokens = tokenizer.collect::<Result<Vec<_>, _>>().unwrap();
//!
//! assert_eq!(tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
//!            ["io", ":", "format", "(", r#""Hello""#, ")", "."]);
//! ```
//!
//! # References
//!
//! - [`erl_scan`][erl_scan] module
//! - [Erlang Data Types][Data Types]
//!
//! [erl_scan]: http://erlang.org/doc/man/erl_scan.html
//! [Data Types]: http://erlang.org/doc/reference_manual/data_types.html
//!
pub mod tokens;
pub mod values;

mod error;
mod hidden_token;
mod lexer;
mod lexical_token;
mod position;
mod token;
mod tokenizer;
mod util;

#[cfg(test)]
mod test;

pub use self::error::{Error, ErrorKind};
pub use self::hidden_token::HiddenToken;
pub use self::lexer::Lexer;
pub use self::lexical_token::LexicalToken;
pub use self::position::{Position, PositionRange};
pub use self::token::Token;
pub use self::tokenizer::Tokenizer;

/// This crate specific `Result` type.
pub type Result<T> = ::std::result::Result<T, Error>;
