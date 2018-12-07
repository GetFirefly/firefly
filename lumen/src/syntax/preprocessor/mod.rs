//! Erlang source code preprocessor.
//!
//! # Examples
//!
//!     use ::syntax::preprocessor::Preprocessor;
//!     use ::syntax::tokenizer::Lexer;
//!
//!     let src = r#"-define(FOO(A), {A, ?LINE}). io:format("Hello: ~p", [?FOO(bar)])."#;
//!     let pp = Preprocessor::new(Lexer::new(src));
//!     let tokens = pp.collect::<Result<Vec<_>, _>>().unwrap();
//!
//!     assert_eq!(tokens.iter().map(|t| t.text()).collect::<Vec<_>>(),
//!                ["io", ":", "format", "(", r#""Hello: ~p""#, ",",
//!                 "[", "{", "bar", ",", "1", "}", "]", ")", "."]);
//!
//! # References
//!
//! - [Erlang Reference Manual -- Preprocessor](http://erlang.org/doc/reference_manual/macros.html)
//!
pub use self::directive::Directive;
pub use self::error::{Error, ErrorKind};
pub use self::macros::{MacroCall, MacroDef};
pub use self::preprocessor::Preprocessor;

pub mod directives;
pub mod types;

mod directive;
mod error;
mod macros;
mod preprocessor;
mod token_reader;
mod util;

#[cfg(test)]
mod test;

/// This crate specific `Result` type.
pub type Result<T> = ::std::result::Result<T, Error>;
