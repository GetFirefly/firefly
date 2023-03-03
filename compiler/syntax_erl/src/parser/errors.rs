use firefly_diagnostics::*;
use firefly_parser::SourceError;

use crate::lexer::Token;
use crate::preprocessor::PreprocessorError;

pub type ParseError = lalrpop_util::ParseError<SourceIndex, Token, ParserError>;

#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    #[error("error reading {path:?}: {source}")]
    RootFile {
        source: std::io::Error,
        path: std::path::PathBuf,
    },

    #[error(transparent)]
    Preprocessor {
        #[from]
        source: PreprocessorError,
    },

    #[error(transparent)]
    Source {
        #[from]
        source: SourceError,
    },

    #[error("{}", .diagnostic.message)]
    ShowDiagnostic { diagnostic: Diagnostic },

    #[error("invalid token")]
    InvalidToken { location: SourceIndex },

    #[error("unrecognized token")]
    UnrecognizedToken {
        span: SourceSpan,
        expected: Vec<String>,
    },

    #[error("extra token")]
    ExtraToken { span: SourceSpan },

    #[error("unexpected eof")]
    UnexpectedEOF {
        location: SourceIndex,
        expected: Vec<String>,
    },
}

impl From<Diagnostic> for ParserError {
    fn from(err: Diagnostic) -> Self {
        ParserError::ShowDiagnostic { diagnostic: err }
    }
}

impl From<ParseError> for ParserError {
    fn from(err: ParseError) -> Self {
        use lalrpop_util::ParseError::*;
        match err {
            InvalidToken { location } => Self::InvalidToken { location },
            UnrecognizedEOF { location, expected } => Self::UnexpectedEOF { location, expected },
            UnrecognizedToken {
                token: (l, _, r),
                expected,
            } => Self::UnrecognizedToken {
                span: SourceSpan::new(l, r),
                expected,
            },
            ExtraToken { token: (l, _, r) } => Self::ExtraToken {
                span: SourceSpan::new(l, r),
            },
            User { .. } => panic!(),
        }
    }
}

impl ToDiagnostic for ParserError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            Self::RootFile { .. } => Diagnostic::error().with_message(self.to_string()),
            Self::ShowDiagnostic { diagnostic } => diagnostic.clone(),
            Self::Preprocessor { source } => source.to_diagnostic(),
            Self::Source { source } => source.to_diagnostic(),
            Self::UnrecognizedToken { span, ref expected } => Diagnostic::error()
                .with_message("unrecognized token")
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message(format!("expected: {}", expected.join(", ")))]),
            Self::InvalidToken { location } => {
                let index = location;
                Diagnostic::error()
                    .with_message("unexpected token")
                    .with_labels(vec![Label::primary(
                        index.source_id(),
                        SourceSpan::new(index, index),
                    )
                    .with_message("did not expect this token")])
            }
            Self::UnexpectedEOF {
                location,
                ref expected,
            } => {
                let index = location;
                if index == SourceIndex::UNKNOWN {
                    Diagnostic::error()
                        .with_message("unexpected end of file")
                        .with_notes(vec![format!("expected: {}", expected.join(", "))])
                } else {
                    Diagnostic::error()
                        .with_message("unexpected end of file")
                        .with_labels(vec![Label::primary(
                            index.source_id(),
                            SourceSpan::new(index, index),
                        )
                        .with_message(format!("expected: {}", expected.join(", ")))])
                }
            }
            Self::ExtraToken { span } => Diagnostic::error()
                .with_message("unexpected token")
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message("did not expect this token")]),
        }
    }
}
