use std::hash::{Hash, Hasher};

use failure::Fail;

use liblumen_diagnostics::{ByteIndex, ByteSpan, Diagnostic, Label};

use super::token::{Token, TokenType};

/// An enum of possible errors that can occur during lexing.
#[derive(Fail, Clone, Debug, PartialEq)]
pub enum LexicalError {
    #[fail(display = "{}", reason)]
    InvalidFloat { span: ByteSpan, reason: String },

    #[fail(display = "{}", reason)]
    InvalidRadix { span: ByteSpan, reason: String },

    /// Occurs when a string literal is not closed (e.g. `"this is an unclosed string`)
    /// It is also implicit that hitting this error means we've reached EOF, as we'll scan the
    /// entire input looking for the closing quote
    #[fail(display = "Unclosed string literal")]
    UnclosedString { span: ByteSpan },

    /// Like UnclosedStringLiteral, but for quoted atoms
    #[fail(display = "Unclosed atom literal")]
    UnclosedAtom { span: ByteSpan },

    /// Occurs when an escape sequence is encountered but the code is unsupported or unrecognized
    #[fail(display = "{}", reason)]
    InvalidEscape { span: ByteSpan, reason: String },

    /// Occurs when we encounter an unexpected character
    #[fail(display = "Encountered unexpected character '{}'", found)]
    UnexpectedCharacter { start: ByteIndex, found: char },
}
impl Hash for LexicalError {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let id = match *self {
            LexicalError::InvalidFloat { .. } => 0,
            LexicalError::InvalidRadix { .. } => 1,
            LexicalError::UnclosedString { .. } => 2,
            LexicalError::UnclosedAtom { .. } => 3,
            LexicalError::InvalidEscape { .. } => 4,
            LexicalError::UnexpectedCharacter { .. } => 5,
        };
        id.hash(state);
    }
}
impl LexicalError {
    /// Return the source span for this error
    pub fn span(&self) -> ByteSpan {
        match *self {
            LexicalError::InvalidFloat { span, .. } => span,
            LexicalError::InvalidRadix { span, .. } => span,
            LexicalError::UnclosedString { span, .. } => span,
            LexicalError::UnclosedAtom { span, .. } => span,
            LexicalError::InvalidEscape { span, .. } => span,
            LexicalError::UnexpectedCharacter { start, .. } => ByteSpan::new(start, start),
        }
    }

    /// Get diagnostic for display
    pub fn to_diagnostic(&self) -> Diagnostic {
        let span = self.span();
        let msg = self.to_string();
        match *self {
            LexicalError::InvalidFloat { .. } => Diagnostic::new_error("invalid float literal")
                .with_label(Label::new_primary(span).with_message(msg)),
            LexicalError::InvalidRadix { .. } => {
                Diagnostic::new_error("invalid radix value for integer literal")
                    .with_label(Label::new_primary(span).with_message(msg))
            }
            LexicalError::InvalidEscape { .. } => Diagnostic::new_error("invalid escape sequence")
                .with_label(Label::new_primary(span).with_message(msg)),
            LexicalError::UnexpectedCharacter { .. } => {
                Diagnostic::new_error("unexpected character")
                    .with_label(Label::new_primary(span).with_message(msg))
            }
            _ => Diagnostic::new_error(msg).with_label(Label::new_primary(span)),
        }
    }
}

/// Produced when converting from LexicalToken to {Atom,Ident,String,Symbol}Token
#[derive(Fail, Debug, Clone)]
#[fail(display = "expected token of {}, got {}", expected, token)]
pub struct TokenConvertError {
    pub span: ByteSpan,
    pub token: Token,
    pub expected: TokenType,
}
