//! A fairly basic lexer for Erlang

mod errors;
mod lexer;
mod scanner;
mod source;
mod symbol;
mod token;

pub use self::errors::{LexicalError, TokenConvertError};
pub use self::lexer::Lexer;
pub use self::scanner::Scanner;
pub use self::source::{FileMapSource, Source, SourceError};
pub use self::symbol::{symbols, SYMBOL_TABLE};
pub use self::symbol::{Ident, InternedString, LocalInternedString, Symbol};
pub use self::token::{AtomToken, IdentToken, StringToken, SymbolToken, TokenType};
pub use self::token::{LexicalToken, Token};

/// The type that serves as an `Item` for the lexer iterator.
pub type Lexed = Result<LexicalToken, LexicalError>;

/// The result type produced by TryFrom<LexicalToken>
pub type TokenConvertResult<T> = Result<T, TokenConvertError>;
