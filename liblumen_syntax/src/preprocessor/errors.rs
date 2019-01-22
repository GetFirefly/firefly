use std;

use failure::{Error, Fail};
use glob;
use itertools::Itertools;
use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};

use crate::lexer::SourceError;
use crate::lexer::{LexicalError, LexicalToken, TokenConvertError};

use crate::parser::ParserError;

use super::directive::Directive;
use super::macros::{MacroCall, MacroDef, Stringify};

#[derive(Fail, Debug)]
pub enum PreprocessorError {
    #[fail(display = "{}", _0)]
    Lexical(#[fail(cause)] LexicalError),

    #[fail(display = "{}", _0)]
    Source(#[fail(cause)] SourceError),

    #[fail(display = "unable to parse constant expression")]
    ParseError(ByteSpan, Vec<ParserError>),

    #[fail(display = "{}", _1)]
    CompilerError(Option<ByteSpan>, String),

    #[fail(display = "invalid constant expression found in preprocessor directive")]
    InvalidConstExpression(ByteSpan),

    #[fail(display = "invalid conditional expression")]
    InvalidConditional(ByteSpan),

    #[fail(display = "call to builtin function failed")]
    BuiltinFailed(ByteSpan, #[fail(cause)] Error),

    #[fail(display = "found orphaned '-end.' directive")]
    OrphanedEnd(Directive),

    #[fail(display = "found orphaned '-else.' directive")]
    OrphanedElse(Directive),

    #[fail(display = "undefined macro")]
    UndefinedStringifyMacro(Stringify),

    #[fail(display = "undefined macro")]
    UndefinedMacro(MacroCall),

    #[fail(display = "invalid macro invocation")]
    BadMacroCall(MacroCall, MacroDef, String),

    #[fail(display = "i/o failure")]
    IO(#[fail(cause)] std::io::Error),

    #[fail(display = "{}", _0)]
    Diagnostic(Diagnostic),

    #[fail(display = "unexpected token")]
    InvalidTokenType(LexicalToken, String),

    #[fail(display = "unexpected token")]
    UnexpectedToken(LexicalToken, Vec<String>),

    #[fail(display = "unexpected eof")]
    UnexpectedEOF,
}
impl PreprocessorError {
    pub fn span(&self) -> Option<ByteSpan> {
        match *self {
            PreprocessorError::Lexical(ref err) => Some(err.span()),
            PreprocessorError::ParseError(ref span, _) => Some(span.clone()),
            PreprocessorError::CompilerError(Some(ref span), _) => Some(span.clone()),
            PreprocessorError::InvalidConstExpression(ref span) => Some(span.clone()),
            PreprocessorError::InvalidConditional(ref span) => Some(span.clone()),
            PreprocessorError::BuiltinFailed(ref span, _) => Some(span.clone()),
            PreprocessorError::OrphanedEnd(ref dir) => Some(dir.span()),
            PreprocessorError::OrphanedElse(ref dir) => Some(dir.span()),
            PreprocessorError::UndefinedStringifyMacro(ref m) => Some(m.span()),
            PreprocessorError::UndefinedMacro(ref m) => Some(m.span()),
            PreprocessorError::BadMacroCall(ref call, _, _) => Some(call.span()),
            PreprocessorError::InvalidTokenType(ref t, _) => Some(t.span()),
            PreprocessorError::UnexpectedToken(ref t, _) => Some(t.span()),
            _ => None,
        }
    }

    pub fn to_diagnostic(&self) -> Diagnostic {
        let span = self.span();
        let msg = self.to_string();
        match *self {
            PreprocessorError::Diagnostic(ref d) => d.clone(),
            PreprocessorError::Lexical(ref err) => err.to_diagnostic(),
            PreprocessorError::Source(ref err) => err.to_diagnostic(),
            PreprocessorError::ParseError(_, ref errs) => {
                let mut d = Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message("invalid constant expression"));
                for err in errs.iter() {
                    let err = err.to_diagnostic();
                    for label in err.labels {
                        d = d.with_label(label);
                    }
                }
                d
            }
            PreprocessorError::CompilerError(Some(_), _) =>
                Diagnostic::new_error("found error directive")
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message(msg)),
            PreprocessorError::InvalidConstExpression(_) =>
                Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message("expected valid constant expression (example: `?OTP_VERSION >= 21`)")),
            PreprocessorError::InvalidConditional(_) =>
                Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message("expected 'true', 'false', or an expression which can be evaluated to 'true' or 'false'")),
            PreprocessorError::BuiltinFailed(_, ref cause) =>
                Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message(cause.to_string())),
            PreprocessorError::BadMacroCall(_, MacroDef::String(_), ref reason) =>
                Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message(reason.to_owned())),
            PreprocessorError::BadMacroCall(_, ref def, ref reason) => {
                let d = Diagnostic::new_error(msg)
                            .with_label(Label::new_primary(span.unwrap())
                                .with_message("this macro call does not match its definition"));
                let secondary_span = match def {
                    MacroDef::Static(ref define) =>
                        define.span(),
                    MacroDef::Dynamic(ref tokens) => {
                        assert!(tokens.len() > 0);
                        ByteSpan::new(
                            tokens[0].span().start(),
                            tokens.last().unwrap().span().end()
                        )
                    },
                    _ => unreachable!()
                };
                d.with_label(Label::new_secondary(secondary_span)
                    .with_message(reason.to_owned()))
            }
            PreprocessorError::IO(_) =>
                Diagnostic::new_error("i/o failed")
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message(msg)),
            PreprocessorError::InvalidTokenType(_, ref expected) =>
                Diagnostic::new_error(msg)
                    .with_label(Label::new_primary(span.unwrap())
                        .with_message(format!("expected \"{}\"", expected))),
            PreprocessorError::UnexpectedToken(_, ref expected) => {
                if expected.len() > 0 {
                    let expected = expected.iter()
                        .map(|t| format!("\"{}\"", t))
                        .join(", ");
                    Diagnostic::new_error(msg)
                        .with_label(Label::new_primary(span.unwrap())
                            .with_message(format!("expected one of {}", expected)))
                } else {
                    Diagnostic::new_error(msg)
                        .with_label(Label::new_primary(span.unwrap()))
                }
            }
            _ if span.is_some() =>
                Diagnostic::new_error("preprocessor error")
                    .with_label(Label::new_primary(span.unwrap()).with_message(msg)),
            _ =>
                Diagnostic::new_error(format!("preprocessor error: {}", msg)),
        }
    }
}
impl From<LexicalError> for PreprocessorError {
    fn from(err: LexicalError) -> PreprocessorError {
        PreprocessorError::Lexical(err)
    }
}
impl From<SourceError> for PreprocessorError {
    fn from(err: SourceError) -> PreprocessorError {
        PreprocessorError::Source(err)
    }
}
impl From<TokenConvertError> for PreprocessorError {
    fn from(err: TokenConvertError) -> PreprocessorError {
        let span = err.span;
        let token = LexicalToken(span.start(), err.token, span.end());
        PreprocessorError::InvalidTokenType(token, err.expected.to_string())
    }
}
impl From<std::io::Error> for PreprocessorError {
    fn from(err: std::io::Error) -> Self {
        PreprocessorError::IO(err)
    }
}
impl From<glob::GlobError> for PreprocessorError {
    fn from(err: glob::GlobError) -> Self {
        PreprocessorError::Diagnostic(Diagnostic::new_error(err.to_string()))
    }
}
impl From<glob::PatternError> for PreprocessorError {
    fn from(err: glob::PatternError) -> Self {
        PreprocessorError::Diagnostic(Diagnostic::new_error(err.to_string()))
    }
}
