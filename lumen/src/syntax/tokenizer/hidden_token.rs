use super::tokens::{CommentToken, WhitespaceToken};
use super::{Position, PositionRange};

/// Hidden token.
///
/// "Hidden" means it has no lexical meanings.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum HiddenToken {
    Comment(CommentToken),
    Whitespace(WhitespaceToken),
}
impl HiddenToken {
    /// Returns the original textual representation of this token.
    pub fn text(&self) -> &str {
        match *self {
            HiddenToken::Comment(ref t) => t.text(),
            HiddenToken::Whitespace(ref t) => t.text(),
        }
    }

    /// Tries to return the reference to the inner `CommentToken`.
    pub fn as_comment_token(&self) -> Option<&CommentToken> {
        if let HiddenToken::Comment(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `WhitespaceToken`.
    pub fn as_whitespace_token(&self) -> Option<&WhitespaceToken> {
        if let HiddenToken::Whitespace(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the inner `CommentToken`.
    pub fn into_comment_token(self) -> Result<CommentToken, Self> {
        if let HiddenToken::Comment(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `WhitespaceToken`.
    pub fn into_whitespace_token(self) -> Result<WhitespaceToken, Self> {
        if let HiddenToken::Whitespace(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }
}
impl From<CommentToken> for HiddenToken {
    fn from(f: CommentToken) -> Self {
        HiddenToken::Comment(f)
    }
}
impl From<WhitespaceToken> for HiddenToken {
    fn from(f: WhitespaceToken) -> Self {
        HiddenToken::Whitespace(f)
    }
}
impl PositionRange for HiddenToken {
    fn start_position(&self) -> Position {
        match *self {
            HiddenToken::Comment(ref t) => t.start_position(),
            HiddenToken::Whitespace(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            HiddenToken::Comment(ref t) => t.end_position(),
            HiddenToken::Whitespace(ref t) => t.end_position(),
        }
    }
}
impl std::fmt::Display for HiddenToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.text().fmt(f)
    }
}
