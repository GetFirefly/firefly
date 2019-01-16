use std::fmt;
use std::mem;
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

use num::BigInt;

use liblumen_diagnostics::{ByteSpan, ByteIndex};

use super::{LexicalError, TokenConvertResult, TokenConvertError, Symbol};

#[derive(Debug, Clone, PartialEq)]
pub struct LexicalToken(pub ByteIndex, pub Token, pub ByteIndex);
impl LexicalToken {
    #[inline]
    pub fn token(&self) -> Token {
        self.1.clone()
    }

    #[inline]
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.0, self.2)
    }
}
impl fmt::Display for LexicalToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}
impl std::convert::Into<(ByteIndex, Token, ByteIndex)> for LexicalToken {
    fn into(self) -> (ByteIndex, Token, ByteIndex) {
        (self.0, self.1, self.2)
    }
}
impl std::convert::From<(ByteIndex, Token, ByteIndex)> for LexicalToken {
    fn from(triple: (ByteIndex, Token, ByteIndex)) -> LexicalToken {
        LexicalToken(triple.0, triple.1, triple.2)
    }
}

/// Used to identify the type of token expected in a TokenConvertError
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenType {
    Atom,
    Ident,
    String,
    Symbol
}
impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TokenType::Atom => write!(f, "ATOM"),
            TokenType::Ident => write!(f, "IDENT"),
            TokenType::String => write!(f, "STRING"),
            TokenType::Symbol => write!(f, "SYMBOL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct AtomToken(ByteIndex, Token, ByteIndex);
impl AtomToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::Atom(a) => a,
            _ => unreachable!()
        }
    }
}
impl TryFrom<LexicalToken> for AtomToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<AtomToken> {
        if let LexicalToken(start, tok @ Token::Atom(_), end) = t {
            return Ok(AtomToken(start, tok, end));
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::Atom,
        })
    }
}
impl Into<LexicalToken> for AtomToken {
    fn into(self) -> LexicalToken {
        LexicalToken(self.0, self.1, self.2)
    }
}
impl fmt::Display for AtomToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct IdentToken(pub ByteIndex, pub Token, pub ByteIndex);
impl IdentToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::Ident(a) => a,
            _ => unreachable!()
        }
    }
}
impl TryFrom<LexicalToken> for IdentToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<IdentToken> {
        if let LexicalToken(start, tok @ Token::Ident(_), end) = t {
            return Ok(IdentToken(start, tok, end));
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::Ident,
        })
    }
}
impl Into<LexicalToken> for IdentToken {
    fn into(self) -> LexicalToken {
        LexicalToken(self.0, self.1, self.2)
    }
}
impl fmt::Display for IdentToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct StringToken(pub ByteIndex, pub Token, pub ByteIndex);
impl StringToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::String(a) => a,
            _ => unreachable!()
        }
    }
}
impl TryFrom<LexicalToken> for StringToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<StringToken> {
        if let LexicalToken(start, tok @ Token::String(_), end) = t {
            return Ok(StringToken(start, tok, end));
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::String,
        })
    }
}
impl Into<LexicalToken> for StringToken {
    fn into(self) -> LexicalToken {
        LexicalToken(self.0, self.1, self.2)
    }
}
impl fmt::Display for StringToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct SymbolToken(pub ByteIndex, pub Token, pub ByteIndex);
impl SymbolToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self.0, self.2)
    }
}
impl TryFrom<LexicalToken> for SymbolToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<SymbolToken> {
        match t {
            LexicalToken(_, Token::Atom(_), _) => (),
            LexicalToken(_, Token::Ident(_), _) => (),
            LexicalToken(_, Token::String(_), _) => (),
            LexicalToken(start, token, end) =>
                return Ok(SymbolToken(start, token, end))
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::Symbol
        })
    }
}
impl Into<LexicalToken> for SymbolToken {
    fn into(self) -> LexicalToken {
        LexicalToken(self.0, self.1, self.2)
    }
}
impl fmt::Display for SymbolToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.fmt(f)
    }
}

/// This enum contains tokens produced by the lexer
#[derive(Debug, Clone)]
pub enum Token {
    // Signifies end of input
    EOF,
    // A tokenization error which may be recovered from
    Error(LexicalError),
    // Docs
    Comment,
    Edoc,
    // Literals
    Char(char),
    Integer(i64),
    BigInteger(BigInt),
    Float(f64),
    Atom(Symbol),
    String(Symbol),
    Ident(Symbol),
    // Keywords and Symbols
    LParen,
    RParen,
    Comma,
    RightStab,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Bar,
    BarBar,
    LeftStab,
    Semicolon,
    Colon,
    Pound,
    Dot,
    After,
    Begin,
    Case,
    Try,
    Catch,
    End,
    Fun,
    If,
    Of,
    Receive,
    When,
    AndAlso,
    OrElse,
    Bnot,
    Not,
    Star,
    Slash,
    Div,
    Rem,
    Band,
    And,
    Plus,
    Minus,
    Bor,
    Bxor,
    Bsl,
    Bsr,
    Or,
    Xor,
    PlusPlus,
    MinusMinus,
    // ==
    IsEqual,
    // /=
    IsNotEqual,
    // =<
    IsLessThanOrEqual,
    // <
    IsLessThan,
    // >=
    IsGreaterThanOrEqual,
    // >
    IsGreaterThan,
    // =:=
    IsExactlyEqual,
    // =/=
    IsExactlyNotEqual,
    // <=
    LeftArrow,
    // =>
    RightArrow,
    // :=
    ColonEqual,
    // <<
    BinaryStart,
    // >>
    BinaryEnd,
    Bang,
    // =
    Equals,
    ColonColon,
    DotDot,
    DotDotDot,
    Question,
    DoubleQuestion,
}
impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        match *self {
            Token::Char(c) =>
                if let Token::Char(c2) = other {
                    return c == *c2;
                },
            Token::Integer(i) =>
                if let Token::Integer(i2) = other {
                    return i == *i2;
                },
            Token::Float(n) =>
                if let Token::Float(n2) = other {
                    return n == *n2;
                },
            Token::Error(_) =>
                if let Token::Error(_) = other {
                    return true;
                },
            Token::Atom(ref a) =>
                if let Token::Atom(a2) = other {
                    return *a == *a2;
                },
            Token::Ident(ref i) =>
                if let Token::Ident(i2) = other {
                    return *i == *i2;
                },
            Token::String(ref s) =>
                if let Token::String(s2) = other {
                    return *s == *s2;
                },
            _ =>
                return mem::discriminant(self) == mem::discriminant(other)
        }
        return false;
    }
}
impl Eq for Token {}
impl Hash for Token {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Token::Float(n) => (n as u64).hash(state),
            Token::Error(ref e) => e.hash(state),
            Token::Atom(ref a) => a.hash(state),
            Token::Ident(ref i) => i.hash(state),
            Token::String(ref s) => s.hash(state),
            Token::Char(c) => c.hash(state),
            ref token => token.to_string().hash(state),
        }
    }
}

impl Token {
    pub fn from_bare_atom<'input>(atom: &'input str) -> Self {
        match atom.as_ref() {
            // Reserved words
            "after" => Token::After,
            "begin" => Token::Begin,
            "case" => Token::Case,
            "try" => Token::Try,
            "catch" => Token::Catch,
            "end" => Token::End,
            "fun" => Token::Fun,
            "if" => Token::If,
            "of" => Token::Of,
            "receive" => Token::Receive,
            "when" => Token::When,
            "andalso" => Token::AndAlso,
            "orelse" => Token::OrElse,
            "bnot" => Token::Bnot,
            "not" => Token::Not,
            "div" => Token::Div,
            "rem" => Token::Rem,
            "band" => Token::Band,
            "and" => Token::And,
            "bor" => Token::Bor,
            "bxor" => Token::Bxor,
            "bsl" => Token::Bsl,
            "bsr" => Token::Bsr,
            "or" => Token::Or,
            "xor" => Token::Xor,
            _ => Token::Atom(Symbol::intern(atom))
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::EOF => write!(f, "EOF"),
            Token::Error(_) => write!(f, "ERROR"),
            Token::Comment => write!(f, "COMMENT"),
            Token::Edoc => write!(f, "EDOC"),
            // Literals
            Token::Char(ref c) => write!(f, "{}", c),
            Token::Integer(ref i) => write!(f, "{}", i),
            Token::BigInteger(ref i) => write!(f, "{}", i),
            Token::Float(ref n) => write!(f, "{}", n),
            Token::Atom(ref s) => write!(f, "'{}'", s),
            Token::String(ref s) => write!(f, "\"{}\"", s),
            Token::Ident(ref s) => write!(f, "{}", s),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Comma => write!(f, ","),
            Token::RightStab => write!(f, "->"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Bar => write!(f, "|"),
            Token::BarBar => write!(f, "||"),
            Token::LeftStab => write!(f, "<-"),
            Token::Semicolon => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Pound => write!(f, "#"),
            Token::Dot => write!(f, "."),
            Token::After => write!(f, "after"),
            Token::Begin => write!(f, "begin"),
            Token::Case => write!(f, "case"),
            Token::Try => write!(f, "try"),
            Token::Catch => write!(f, "catch"),
            Token::End => write!(f, "end"),
            Token::Fun => write!(f, "fun"),
            Token::If => write!(f, "if"),
            Token::Of => write!(f, "of"),
            Token::Receive => write!(f, "receive"),
            Token::When => write!(f, "when"),
            Token::AndAlso => write!(f, "andalso"),
            Token::OrElse => write!(f, "orelse"),
            Token::Bnot => write!(f, "bnot"),
            Token::Not => write!(f, "not"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Div => write!(f, "div"),
            Token::Rem => write!(f, "rem"),
            Token::Band => write!(f, "band"),
            Token::And => write!(f, "and"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Bor => write!(f, "bor"),
            Token::Bxor => write!(f, "bxor"),
            Token::Bsl => write!(f, "bsl"),
            Token::Bsr => write!(f, "bsr"),
            Token::Or => write!(f, "or"),
            Token::Xor => write!(f, "xor"),
            Token::PlusPlus => write!(f, "++"),
            Token::MinusMinus => write!(f, "--"),
            Token::IsEqual => write!(f, "=="),
            Token::IsNotEqual => write!(f, "/="),
            Token::IsLessThanOrEqual => write!(f, "=<"),
            Token::IsLessThan => write!(f, "<"),
            Token::IsGreaterThanOrEqual => write!(f, ">="),
            Token::IsGreaterThan => write!(f, ">"),
            Token::IsExactlyEqual => write!(f, "=:="),
            Token::IsExactlyNotEqual => write!(f, "=/="),
            Token::LeftArrow => write!(f, "<="),
            Token::RightArrow => write!(f, "=>"),
            Token::ColonEqual => write!(f, ":="),
            Token::BinaryStart => write!(f, "<<"),
            Token::BinaryEnd => write!(f, ">>"),
            Token::Bang => write!(f, "!"),
            Token::Equals => write!(f, "="),
            Token::ColonColon => write!(f, "::"),
            Token::DotDot => write!(f, ".."),
            Token::DotDotDot => write!(f, "..."),
            Token::Question => write!(f, "?"),
            Token::DoubleQuestion => write!(f, "??"),
        }
    }
}
