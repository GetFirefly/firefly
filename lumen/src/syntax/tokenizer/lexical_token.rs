use super::tokens::*;
use super::{Position, PositionRange};

/// Lexical token.
///
/// This kind of token is meaningful in lexical analysis.
#[derive(Debug, Clone)]
pub enum LexicalToken {
    Atom(AtomToken),
    Char(CharToken),
    Float(FloatToken),
    Integer(IntegerToken),
    Keyword(KeywordToken),
    String(StringToken),
    Symbol(SymbolToken),
    Variable(VariableToken),
}
impl LexicalToken {
    /// Returns the original textual representation of this token.
    pub fn text(&self) -> &str {
        match *self {
            LexicalToken::Atom(ref t) => t.text(),
            LexicalToken::Char(ref t) => t.text(),
            LexicalToken::Float(ref t) => t.text(),
            LexicalToken::Integer(ref t) => t.text(),
            LexicalToken::Keyword(ref t) => t.text(),
            LexicalToken::String(ref t) => t.text(),
            LexicalToken::Symbol(ref t) => t.text(),
            LexicalToken::Variable(ref t) => t.text(),
        }
    }

    /// Tries to return the reference to the inner `AtomToken`.
    pub fn as_atom_token(&self) -> Option<&AtomToken> {
        if let LexicalToken::Atom(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `CharToken`.
    pub fn as_char_token(&self) -> Option<&CharToken> {
        if let LexicalToken::Char(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `FloatToken`.
    pub fn as_float_token(&self) -> Option<&FloatToken> {
        if let LexicalToken::Float(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `IntegerToken`.
    pub fn as_integer_token(&self) -> Option<&IntegerToken> {
        if let LexicalToken::Integer(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `KeywordToken`.
    pub fn as_keyword_token(&self) -> Option<&KeywordToken> {
        if let LexicalToken::Keyword(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `StringToken`.
    pub fn as_string_token(&self) -> Option<&StringToken> {
        if let LexicalToken::String(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `SymbolToken`.
    pub fn as_symbol_token(&self) -> Option<&SymbolToken> {
        if let LexicalToken::Symbol(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the reference to the inner `VariableToken`.
    pub fn as_variable_token(&self) -> Option<&VariableToken> {
        if let LexicalToken::Variable(ref t) = *self {
            Some(t)
        } else {
            None
        }
    }

    /// Tries to return the inner `AtomToken`.
    pub fn into_atom_token(self) -> Result<AtomToken, Self> {
        if let LexicalToken::Atom(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `CharToken`.
    pub fn into_char_token(self) -> Result<CharToken, Self> {
        if let LexicalToken::Char(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `FloatToken`.
    pub fn into_float_token(self) -> Result<FloatToken, Self> {
        if let LexicalToken::Float(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `IntegerToken`.
    pub fn into_integer_token(self) -> Result<IntegerToken, Self> {
        if let LexicalToken::Integer(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `KeywordToken`.
    pub fn into_keyword_token(self) -> Result<KeywordToken, Self> {
        if let LexicalToken::Keyword(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `StringToken`.
    pub fn into_string_token(self) -> Result<StringToken, Self> {
        if let LexicalToken::String(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `SymbolToken`.
    pub fn into_symbol_token(self) -> Result<SymbolToken, Self> {
        if let LexicalToken::Symbol(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }

    /// Tries to return the inner `VariableToken`.
    pub fn into_variable_token(self) -> Result<VariableToken, Self> {
        if let LexicalToken::Variable(t) = self {
            Ok(t)
        } else {
            Err(self)
        }
    }
}
impl From<AtomToken> for LexicalToken {
    fn from(f: AtomToken) -> Self {
        LexicalToken::Atom(f)
    }
}
impl From<CharToken> for LexicalToken {
    fn from(f: CharToken) -> Self {
        LexicalToken::Char(f)
    }
}
impl From<FloatToken> for LexicalToken {
    fn from(f: FloatToken) -> Self {
        LexicalToken::Float(f)
    }
}
impl From<IntegerToken> for LexicalToken {
    fn from(f: IntegerToken) -> Self {
        LexicalToken::Integer(f)
    }
}
impl From<KeywordToken> for LexicalToken {
    fn from(f: KeywordToken) -> Self {
        LexicalToken::Keyword(f)
    }
}
impl From<StringToken> for LexicalToken {
    fn from(f: StringToken) -> Self {
        LexicalToken::String(f)
    }
}
impl From<SymbolToken> for LexicalToken {
    fn from(f: SymbolToken) -> Self {
        LexicalToken::Symbol(f)
    }
}
impl From<VariableToken> for LexicalToken {
    fn from(f: VariableToken) -> Self {
        LexicalToken::Variable(f)
    }
}
impl PositionRange for LexicalToken {
    fn start_position(&self) -> Position {
        match *self {
            LexicalToken::Atom(ref t) => t.start_position(),
            LexicalToken::Char(ref t) => t.start_position(),
            LexicalToken::Float(ref t) => t.start_position(),
            LexicalToken::Integer(ref t) => t.start_position(),
            LexicalToken::Keyword(ref t) => t.start_position(),
            LexicalToken::String(ref t) => t.start_position(),
            LexicalToken::Symbol(ref t) => t.start_position(),
            LexicalToken::Variable(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            LexicalToken::Atom(ref t) => t.end_position(),
            LexicalToken::Char(ref t) => t.end_position(),
            LexicalToken::Float(ref t) => t.end_position(),
            LexicalToken::Integer(ref t) => t.end_position(),
            LexicalToken::Keyword(ref t) => t.end_position(),
            LexicalToken::String(ref t) => t.end_position(),
            LexicalToken::Symbol(ref t) => t.end_position(),
            LexicalToken::Variable(ref t) => t.end_position(),
        }
    }
}
impl std::fmt::Display for LexicalToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.text().fmt(f)
    }
}
