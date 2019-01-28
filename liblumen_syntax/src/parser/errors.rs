use std::convert::From;

use failure::Fail;
use liblumen_diagnostics::{ByteIndex, ByteSpan, Diagnostic, Label};

use crate::lexer::{SourceError, Token};
use crate::preprocessor::PreprocessorError;

pub type ParseError = lalrpop_util::ParseError<ByteIndex, Token, ParserError>;

#[derive(Fail, Debug)]
pub enum ParserError {
    #[fail(display = "{}", _0)]
    Preprocessor(#[fail(cause)] PreprocessorError),

    #[fail(display = "{}", _0)]
    Source(#[fail(cause)] SourceError),

    #[fail(display = "i/o error")]
    IO(#[fail(cause)] std::io::Error),

    #[fail(display = "{}", _0)]
    Diagnostic(Diagnostic),

    #[fail(display = "invalid token")]
    InvalidToken { span: ByteSpan },

    #[fail(display = "unrecognized token")]
    UnrecognizedToken {
        span: ByteSpan,
        expected: Vec<String>,
    },

    #[fail(display = "extra token")]
    ExtraToken { span: ByteSpan },

    #[fail(display = "unexpected eof")]
    UnexpectedEOF { expected: Vec<String> },
}
impl From<ParseError> for ParserError {
    fn from(err: ParseError) -> ParserError {
        match err {
            lalrpop_util::ParseError::InvalidToken { location } => ParserError::InvalidToken {
                span: ByteSpan::new(location, location),
            },
            lalrpop_util::ParseError::UnrecognizedToken {
                token: None,
                expected,
            } => ParserError::UnexpectedEOF { expected },
            lalrpop_util::ParseError::UnrecognizedToken {
                token: Some((start, _, end)),
                expected,
            } => ParserError::UnrecognizedToken {
                span: ByteSpan::new(start, end),
                expected,
            },
            lalrpop_util::ParseError::ExtraToken {
                token: (start, _, end),
            } => ParserError::ExtraToken {
                span: ByteSpan::new(start, end),
            },
            lalrpop_util::ParseError::User { error: err } => err.into(),
        }
    }
}
impl From<std::io::Error> for ParserError {
    fn from(err: std::io::Error) -> ParserError {
        ParserError::IO(err)
    }
}
impl From<PreprocessorError> for ParserError {
    fn from(err: PreprocessorError) -> ParserError {
        ParserError::Preprocessor(err)
    }
}
impl From<SourceError> for ParserError {
    fn from(err: SourceError) -> ParserError {
        ParserError::Source(err)
    }
}
impl ParserError {
    pub fn span(&self) -> Option<ByteSpan> {
        match self {
            ParserError::Preprocessor(ref err) => err.span(),
            ParserError::InvalidToken { ref span, .. } => Some(span.clone()),
            ParserError::UnrecognizedToken { ref span, .. } => Some(span.clone()),
            ParserError::ExtraToken { ref span, .. } => Some(span.clone()),
            _ => None,
        }
    }

    pub fn to_diagnostic(&self) -> Diagnostic {
        let span = self.span();
        let msg = self.to_string();
        match *self {
            ParserError::Diagnostic(ref d) => d.clone(),
            ParserError::Preprocessor(ref err) => err.to_diagnostic(),
            ParserError::Source(ref err) => err.to_diagnostic(),
            ParserError::IO(_) => Diagnostic::new_error("i/o failed")
                .with_label(Label::new_primary(span.unwrap()).with_message(msg)),
            ParserError::UnrecognizedToken { ref expected, .. } => {
                Diagnostic::new_error(format!("expected: {}", expected.join(", ")))
                    .with_label(Label::new_primary(span.unwrap()).with_message(msg))
            }
            _ if span.is_some() => {
                Diagnostic::new_error(msg).with_label(Label::new_primary(span.unwrap()))
            }
            _ => Diagnostic::new_error(msg),
        }
    }
}
