use std;
use std::path::PathBuf;

use itertools::Itertools;
use thiserror::Error;

use firefly_diagnostics::*;
use firefly_intern::Symbol;
use firefly_parser::SourceError;

use crate::lexer::{LexicalError, LexicalToken, TokenConvertError};
use crate::parser::ParserError;

use super::directive::Directive;
use super::directives::DirectiveError;
use super::macros::{MacroCall, MacroDef, Stringify};

#[derive(Debug, Error)]
pub enum PreprocessorError {
    #[error(transparent)]
    Lexical {
        #[from]
        source: LexicalError,
    },

    #[error(transparent)]
    Source {
        #[from]
        source: SourceError,
    },

    #[error("error occurred while including {path:?}")]
    IncludeError {
        source: std::io::Error,
        path: PathBuf,
        span: SourceSpan,
    },

    #[error("unable to parse constant expression")]
    ParseError {
        span: SourceSpan,
        inner: Box<ParserError>,
    },

    #[error("{reason}")]
    CompilerError {
        span: Option<SourceSpan>,
        reason: String,
    },

    #[error("invalid constant expression found in preprocessor directive")]
    InvalidConstExpression { span: SourceSpan },

    #[error(transparent)]
    EvalError {
        #[from]
        source: crate::evaluator::EvalError,
    },

    #[error(transparent)]
    BadDirective {
        #[from]
        source: DirectiveError,
    },

    #[error("invalid conditional expression")]
    InvalidConditional { span: SourceSpan },

    #[error("call to builtin function failed")]
    BuiltinFailed {
        span: SourceSpan,
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("found orphaned '-end.' directive")]
    OrphanedEnd { directive: Directive },

    #[error("found orphaned '-else.' directive")]
    OrphanedElse { directive: Directive },

    #[error("undefined macro")]
    UndefinedStringifyMacro { call: Stringify },

    #[error("undefined macro")]
    UndefinedMacro { call: MacroCall },

    #[error("invalid macro invocation")]
    BadMacroCall {
        call: MacroCall,
        def: MacroDef,
        reason: String,
    },

    #[error("{}", .diagnostic.message)]
    ShowDiagnostic { diagnostic: Diagnostic },

    #[error("unexpected token")]
    InvalidTokenType {
        token: LexicalToken,
        expected: String,
    },

    #[error("unexpected token")]
    UnexpectedToken {
        token: LexicalToken,
        expected: Vec<String>,
    },

    #[error("unexpected eof")]
    UnexpectedEOF,

    #[error("warning: {message}")]
    WarningDirective {
        span: SourceSpan,
        message: Symbol,
        as_error: bool,
    },
}
impl ToDiagnostic for PreprocessorError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            PreprocessorError::Lexical { source } => source.to_diagnostic(),
            PreprocessorError::Source { source } => source.to_diagnostic(),
            PreprocessorError::IncludeError { span, .. } => {
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                        .with_message("while processing include directive"),
                    ])
            },
            PreprocessorError::ParseError { span, inner } => {
                let err = inner.to_diagnostic();
                err.with_labels(vec![
                    Label::secondary(span.source_id(), span)
                        .with_message("parsing of this expression failed when attempting to evaluate it as a constant")
                ])
            }
            PreprocessorError::CompilerError { span: Some(span), reason } =>
                Diagnostic::error()
                    .with_message(reason)
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                    ]),
            PreprocessorError::CompilerError { span: None, reason } =>
                Diagnostic::error().with_message(reason),
            PreprocessorError::InvalidConstExpression { span, .. } =>
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                            .with_message("expected valid constant expression (example: `?OTP_VERSION >= 21`)")
                    ]),
            PreprocessorError::EvalError { source } => source.to_diagnostic(),
            PreprocessorError::BadDirective { source } => source.to_diagnostic(),
            PreprocessorError::InvalidConditional { span, .. } =>
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                            .with_message("expected 'true', 'false', or an expression which can be evaluated to 'true' or 'false'")
                    ]),
            PreprocessorError::BuiltinFailed { span, ref source } =>
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                            .with_message(source.to_string())
                    ]),
            PreprocessorError::OrphanedEnd { ref directive } => {
                let span = directive.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                    ])
            }
            PreprocessorError::OrphanedElse { ref directive } => {
                let span = directive.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                    ])
            }
            PreprocessorError::UndefinedStringifyMacro { ref call } => {
                let span = call.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                    ])
            }
            PreprocessorError::UndefinedMacro { ref call } => {
                let span = call.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                    ])
            }
            PreprocessorError::BadMacroCall { ref call, def: MacroDef::String(_), ref reason, .. } => {
                let span = call.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(span.source_id(), span)
                            .with_message(reason.to_owned())
                    ])
            }
            PreprocessorError::BadMacroCall { ref call, ref def, ref reason, .. } => {
                let secondary_span = match def {
                    MacroDef::Static(ref define) => define.span(),
                    MacroDef::Dynamic(ref tokens) => {
                        assert!(tokens.len() > 0);
                        SourceSpan::new(
                            tokens[0].span().start(),
                            tokens.last().unwrap().span().end()
                        )
                    },
                    _ => unreachable!()
                };
                let call_span = call.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(call_span.source_id(), call_span)
                            .with_message("this macro call does not match its definition"),
                        Label::secondary(secondary_span.source_id(), secondary_span)
                            .with_message(reason.to_owned())
                    ])
            }
            PreprocessorError::ShowDiagnostic { diagnostic } => diagnostic.clone(),
            PreprocessorError::InvalidTokenType { ref token, ref expected } => {
                let token_span = token.span();
                Diagnostic::error()
                    .with_message(self.to_string())
                    .with_labels(vec![
                        Label::primary(token_span.source_id(), token_span)
                            .with_message(format!("expected \"{}\"", expected))
                    ])
            }
            PreprocessorError::UnexpectedToken { ref token, ref expected } => {
                let token_span = token.span();
                if expected.len() > 0 {
                    let expected = expected.iter()
                        .map(|t| format!("\"{}\"", t))
                        .join(", ");
                    Diagnostic::error()
                        .with_message(self.to_string())
                        .with_labels(vec![
                            Label::primary(token_span.source_id(), token_span)
                                .with_message(format!("expected one of {}", expected))
                        ])
                } else {
                    Diagnostic::error()
                        .with_message(self.to_string())
                        .with_labels(vec![
                            Label::primary(token_span.source_id(), token_span)
                        ])
                }
            }
            PreprocessorError::UnexpectedEOF =>
                Diagnostic::error().with_message(self.to_string()),
            PreprocessorError::WarningDirective { span, message, as_error } => {
                let message_str = message.as_str().get();
                if as_error { Diagnostic::error() } else { Diagnostic::warning() }
                    .with_message(message_str)
                    .with_labels(vec![
                        Label::primary(span.source_id(), span),
                    ])
            }
        }
    }
}
impl From<TokenConvertError> for PreprocessorError {
    fn from(err: TokenConvertError) -> PreprocessorError {
        let span = err.span;
        let token = LexicalToken(span.start(), err.token, span.end());
        PreprocessorError::InvalidTokenType {
            token,
            expected: err.expected.to_string(),
        }
    }
}
impl From<Diagnostic> for PreprocessorError {
    fn from(diagnostic: Diagnostic) -> Self {
        PreprocessorError::ShowDiagnostic { diagnostic }
    }
}
