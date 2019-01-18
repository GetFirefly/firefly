mod directive;
mod macros;
mod errors;
mod preprocessor;
mod evaluator;
mod token_stream;
mod token_reader;

pub mod directives;
pub mod types;

pub use self::directive::Directive;
pub use self::macros::{MacroCall, MacroDef};
pub use self::errors::PreprocessorError;
pub use self::preprocessor::Preprocessor;

use liblumen_diagnostics::ByteIndex;

use crate::lexer::Token;

/// The result produced by the preprocessor
pub type Preprocessed = std::result::Result<(ByteIndex, Token, ByteIndex), PreprocessorError>;

type Result<T> = std::result::Result<T, PreprocessorError>;
