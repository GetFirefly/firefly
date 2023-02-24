use std::hash::{Hash, Hasher};

use firefly_diagnostics::{Diagnostic, Label, SourceIndex, SourceSpan, ToDiagnostic};
use firefly_parser::EscapeStmError;

/// An enum of possible errors that can occur during lexing.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum LexicalError {
    #[error("{reason}")]
    InvalidFloat { span: SourceSpan, reason: String },

    #[error("{reason}")]
    InvalidRadix { span: SourceSpan, reason: String },

    /// Occurs when a string literal is not closed (e.g. `"this is an unclosed string`)
    /// It is also implicit that hitting this error means we've reached EOF, as we'll scan the
    /// entire input looking for the closing quote
    #[error("Unclosed string literal")]
    UnclosedString { span: SourceSpan },

    /// Like UnclosedStringLiteral, but for quoted atoms
    #[error("Unclosed atom literal")]
    UnclosedAtom { span: SourceSpan },

    #[error(transparent)]
    EscapeError {
        #[from]
        source: EscapeStmError<SourceIndex>,
    },

    /// Occurs when we encounter an unexpected character
    #[error("Encountered unexpected character '{found}'")]
    UnexpectedCharacter { start: SourceIndex, found: char },
}
impl Hash for LexicalError {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let id = match *self {
            LexicalError::InvalidFloat { .. } => 0,
            LexicalError::InvalidRadix { .. } => 1,
            LexicalError::UnclosedString { .. } => 2,
            LexicalError::UnclosedAtom { .. } => 3,
            LexicalError::EscapeError { .. } => 4,
            LexicalError::UnexpectedCharacter { .. } => 5,
        };
        id.hash(state);
    }
}
impl ToDiagnostic for LexicalError {
    fn to_diagnostic(self) -> Diagnostic {
        let span = self.span();
        let msg = self.to_string();
        match self {
            LexicalError::InvalidFloat { .. } => Diagnostic::error()
                .with_message("invalid float literal")
                .with_labels(vec![
                    Label::primary(span.source_id(), span).with_message(msg)
                ]),
            LexicalError::InvalidRadix { .. } => Diagnostic::error()
                .with_message("invalid radix value for integer literal")
                .with_labels(vec![
                    Label::primary(span.source_id(), span).with_message(msg)
                ]),
            LexicalError::EscapeError { source } => source.to_diagnostic(),
            LexicalError::UnexpectedCharacter { .. } => Diagnostic::error()
                .with_message("unexpected character")
                .with_labels(vec![
                    Label::primary(span.source_id(), span).with_message(msg)
                ]),
            _ => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
        }
    }
}
impl LexicalError {
    /// Return the source span for this error
    pub fn span(&self) -> SourceSpan {
        match self {
            LexicalError::InvalidFloat { span, .. } => *span,
            LexicalError::InvalidRadix { span, .. } => *span,
            LexicalError::UnclosedString { span, .. } => *span,
            LexicalError::UnclosedAtom { span, .. } => *span,
            LexicalError::EscapeError { source } => source.span(),
            LexicalError::UnexpectedCharacter { start, .. } => SourceSpan::new(*start, *start),
        }
    }
}
