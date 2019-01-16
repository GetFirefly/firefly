//! A fairly basic lexer for Erlang

mod errors;
mod scanner;
mod source;
mod token;
mod symbol;
mod lexer;

pub use self::errors::{LexicalError, TokenConvertError};
pub use self::token::{LexicalToken, Token};
pub use self::token::{TokenType, AtomToken, IdentToken, StringToken, SymbolToken};
pub use self::source::{Source, FileMapSource, SourceError};
pub use self::scanner::Scanner;
pub use self::lexer::Lexer;
pub use self::symbol::{symbols, SYMBOL_TABLE};
pub use self::symbol::{Ident, Symbol, InternedString, LocalInternedString};

/// The type that serves as an `Item` for the lexer iterator.
pub type Lexed = Result<LexicalToken, LexicalError>;

/// The result type produced by TryFrom<LexicalToken>
pub type TokenConvertResult<T> = Result<T, TokenConvertError>;
