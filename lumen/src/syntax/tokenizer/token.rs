use trackable::{track, track_assert_some, track_panic};

use super::tokens::*;
use super::{ErrorKind, HiddenToken, LexicalToken, Position, PositionRange};

/// Token.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
pub enum Token {
    Atom(AtomToken),
    Char(CharToken),
    Comment(CommentToken),
    Float(FloatToken),
    Integer(IntegerToken),
    Keyword(KeywordToken),
    String(StringToken),
    Symbol(SymbolToken),
    Variable(VariableToken),
    Whitespace(WhitespaceToken),
}
impl Token {
    /// Tries to convert from any prefixes of the text to a token.
    ///
    /// # Examples
    ///
    /// ```
    /// use erl_tokenize::{Token, Position};
    /// use erl_tokenize::values::Symbol;
    ///
    /// let pos = Position::new();
    ///
    /// // Atom
    /// let token = Token::from_text("foo", pos.clone()).unwrap();
    /// assert_eq!(token.as_atom_token().map(|t| t.value()), Some("foo"));
    ///
    /// // Symbol
    /// let token = Token::from_text("[foo]", pos.clone()).unwrap();
    /// assert_eq!(token.as_symbol_token().map(|t| t.value()), Some(Symbol::OpenSquare));
    /// ```
    pub fn from_text(text: &str, pos: Position) -> super::Result<Self> {
        let head = track_assert_some!(text.chars().nth(0), ErrorKind::UnexpectedEos);
        match head {
            ' ' | '\t' | '\r' | '\n' | '\u{A0}' => {
                track!(WhitespaceToken::from_text(text, pos)).map(Token::from)
            }
            'A'...'Z' | '_' => track!(VariableToken::from_text(text, pos)).map(Token::from),
            '0'...'9' => {
                let maybe_float = if let Some(i) = text.find(|c: char| !c.is_digit(10)) {
                    text.as_bytes()[i] == b'.'
                        && text
                            .as_bytes()
                            .get(i + 1)
                            .map_or(false, |c| (*c as char).is_digit(10))
                } else {
                    false
                };
                if maybe_float {
                    track!(FloatToken::from_text(text, pos)).map(Token::from)
                } else {
                    track!(IntegerToken::from_text(text, pos)).map(Token::from)
                }
            }
            '$' => track!(CharToken::from_text(text, pos)).map(Token::from),
            '"' => track!(StringToken::from_text(text, pos)).map(Token::from),
            '\'' => track!(AtomToken::from_text(text, pos)).map(Token::from),
            '%' => track!(CommentToken::from_text(text, pos)).map(Token::from),
            _ => {
                if head.is_alphabetic() {
                    let atom = track!(AtomToken::from_text(text, pos.clone()))?;
                    if let Ok(keyword) = KeywordToken::from_text(atom.text(), pos) {
                        Ok(Token::from(keyword))
                    } else {
                        Ok(Token::from(atom))
                    }
                } else {
                    track!(SymbolToken::from_text(text, pos)).map(Token::from)
                }
            }
        }
    }

    /// Returns the original textual representation of this token.
    ///
    /// # Examples
    ///
    /// ```
    /// use erl_tokenize::{Token, Position};
    ///
    /// let pos = Position::new();
    ///
    /// // Comment
    /// assert_eq!(Token::from_text("% foo", pos.clone()).unwrap().text(), "% foo");
    ///
    /// // Char
    /// assert_eq!(Token::from_text(r#"$\t"#, pos.clone()).unwrap().text(), r#"$\t"#);
    /// ```
    pub fn text(&self) -> &str {
        match *self {
            Token::Atom(ref t) => t.text(),
            Token::Char(ref t) => t.text(),
            Token::Comment(ref t) => t.text(),
            Token::Float(ref t) => t.text(),
            Token::Integer(ref t) => t.text(),
            Token::Keyword(ref t) => t.text(),
            Token::String(ref t) => t.text(),
            Token::Symbol(ref t) => t.text(),
            Token::Variable(ref t) => t.text(),
            Token::Whitespace(ref t) => t.text(),
        }
    }

    /// Returns `true` if this is a lexical token, otherwise `false`.
    pub fn is_lexical_token(&self) -> bool {
        !self.is_hidden_token()
    }

    /// Returns `true` if this is a hidden token, otherwise `false`.
    pub fn is_hidden_token(&self) -> bool {
        match *self {
            Token::Whitespace(_) | Token::Comment(_) => true,
            _ => false,
        }
    }

    /// Tries to convert into `LexicalToken`.
    pub fn into_lexical_token(self) -> Result<LexicalToken, Self> {
        match self {
            Token::Atom(t) => Ok(t.into()),
            Token::Char(t) => Ok(t.into()),
            Token::Float(t) => Ok(t.into()),
            Token::Integer(t) => Ok(t.into()),
            Token::Keyword(t) => Ok(t.into()),
            Token::String(t) => Ok(t.into()),
            Token::Symbol(t) => Ok(t.into()),
            Token::Variable(t) => Ok(t.into()),
            _ => Err(self),
        }
    }

    /// Tries to convert into `HiddenToken`.
    pub fn into_hidden_token(self) -> Result<HiddenToken, Self> {
        match self {
            Token::Comment(t) => Ok(t.into()),
            Token::Whitespace(t) => Ok(t.into()),
            _ => Err(self),
        }
    }

    /// Tries to return the reference to the inner `AtomToken`.
    pub fn as_atom_token(&self) -> Option<&AtomToken> {
        if let Token::Atom(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `CharToken`.
    pub fn as_char_token(&self) -> Option<&CharToken> {
        if let Token::Char(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `FloatToken`.
    pub fn as_float_token(&self) -> Option<&FloatToken> {
        if let Token::Float(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `IntegerToken`.
    pub fn as_integer_token(&self) -> Option<&IntegerToken> {
        if let Token::Integer(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `KeywordToken`.
    pub fn as_keyword_token(&self) -> Option<&KeywordToken> {
        if let Token::Keyword(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `StringToken`.
    pub fn as_string_token(&self) -> Option<&StringToken> {
        if let Token::String(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `SymbolToken`.
    pub fn as_symbol_token(&self) -> Option<&SymbolToken> {
        if let Token::Symbol(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `VariableToken`.
    pub fn as_variable_token(&self) -> Option<&VariableToken> {
        if let Token::Variable(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `CommentToken`.
    pub fn as_comment_token(&self) -> Option<&CommentToken> {
        if let Token::Comment(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `WhitespaceToken`.
    pub fn as_whitespace_token(&self) -> Option<&WhitespaceToken> {
        if let Token::Whitespace(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the inner `AtomToken`.
    pub fn into_atom_token(self) -> Result<AtomToken, Self> {
        if let Token::Atom(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `CharToken`.
    pub fn into_char_token(self) -> Result<CharToken, Self> {
        if let Token::Char(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `FloatToken`.
    pub fn into_float_token(self) -> Result<FloatToken, Self> {
        if let Token::Float(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `IntegerToken`.
    pub fn into_integer_token(self) -> Result<IntegerToken, Self> {
        if let Token::Integer(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `KeywordToken`.
    pub fn into_keyword_token(self) -> Result<KeywordToken, Self> {
        if let Token::Keyword(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `StringToken`.
    pub fn into_string_token(self) -> Result<StringToken, Self> {
        if let Token::String(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `SymbolToken`.
    pub fn into_symbol_token(self) -> Result<SymbolToken, Self> {
        if let Token::Symbol(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `VariableToken`.
    pub fn into_variable_token(self) -> Result<VariableToken, Self> {
        if let Token::Variable(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `CommentToken`.
    pub fn into_comment_token(self) -> Result<CommentToken, Self> {
        if let Token::Comment(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `WhitespaceToken`.
    pub fn into_whitespace_token(self) -> Result<WhitespaceToken, Self> {
        if let Token::Whitespace(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }
}
impl From<AtomToken> for Token {
    fn from(f: AtomToken) -> Self {
        Token::Atom(f)
    }
}
impl From<CharToken> for Token {
    fn from(f: CharToken) -> Self {
        Token::Char(f)
    }
}
impl From<CommentToken> for Token {
    fn from(f: CommentToken) -> Self {
        Token::Comment(f)
    }
}
impl From<FloatToken> for Token {
    fn from(f: FloatToken) -> Self {
        Token::Float(f)
    }
}
impl From<IntegerToken> for Token {
    fn from(f: IntegerToken) -> Self {
        Token::Integer(f)
    }
}
impl From<KeywordToken> for Token {
    fn from(f: KeywordToken) -> Self {
        Token::Keyword(f)
    }
}
impl From<StringToken> for Token {
    fn from(f: StringToken) -> Self {
        Token::String(f)
    }
}
impl From<SymbolToken> for Token {
    fn from(f: SymbolToken) -> Self {
        Token::Symbol(f)
    }
}
impl From<VariableToken> for Token {
    fn from(f: VariableToken) -> Self {
        Token::Variable(f)
    }
}
impl From<WhitespaceToken> for Token {
    fn from(f: WhitespaceToken) -> Self {
        Token::Whitespace(f)
    }
}
impl From<HiddenToken> for Token {
    fn from(f: HiddenToken) -> Self {
        match f {
            HiddenToken::Comment(t) => t.into(),
            HiddenToken::Whitespace(t) => t.into(),
        }
    }
}
impl From<LexicalToken> for Token {
    fn from(f: LexicalToken) -> Self {
        match f {
            LexicalToken::Atom(t) => t.into(),
            LexicalToken::Char(t) => t.into(),
            LexicalToken::Float(t) => t.into(),
            LexicalToken::Integer(t) => t.into(),
            LexicalToken::Keyword(t) => t.into(),
            LexicalToken::String(t) => t.into(),
            LexicalToken::Symbol(t) => t.into(),
            LexicalToken::Variable(t) => t.into(),
        }
    }
}
impl PositionRange for Token {
    fn start_position(&self) -> Position {
        match *self {
            Token::Atom(ref t) => t.start_position(),
            Token::Char(ref t) => t.start_position(),
            Token::Comment(ref t) => t.start_position(),
            Token::Float(ref t) => t.start_position(),
            Token::Integer(ref t) => t.start_position(),
            Token::Keyword(ref t) => t.start_position(),
            Token::String(ref t) => t.start_position(),
            Token::Symbol(ref t) => t.start_position(),
            Token::Variable(ref t) => t.start_position(),
            Token::Whitespace(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            Token::Atom(ref t) => t.end_position(),
            Token::Char(ref t) => t.end_position(),
            Token::Comment(ref t) => t.end_position(),
            Token::Float(ref t) => t.end_position(),
            Token::Integer(ref t) => t.end_position(),
            Token::Keyword(ref t) => t.end_position(),
            Token::String(ref t) => t.end_position(),
            Token::Symbol(ref t) => t.end_position(),
            Token::Variable(ref t) => t.end_position(),
            Token::Whitespace(ref t) => t.end_position(),
        }
    }
}
impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.text().fmt(f)
    }
}
