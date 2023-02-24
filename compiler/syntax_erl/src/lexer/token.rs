use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use firefly_diagnostics::{SourceIndex, SourceSpan};
use firefly_intern::Symbol;
use firefly_number::{Float, Int, ToPrimitive};

use super::{LexicalError, TokenConvertError, TokenConvertResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexicalToken(pub SourceIndex, pub Token, pub SourceIndex);
impl LexicalToken {
    #[inline]
    pub fn token(&self) -> Token {
        self.1.clone()
    }

    #[inline]
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
}
impl fmt::Display for LexicalToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.token())
    }
}
impl std::convert::Into<(SourceIndex, Token, SourceIndex)> for LexicalToken {
    fn into(self) -> (SourceIndex, Token, SourceIndex) {
        (self.0, self.1, self.2)
    }
}
impl std::convert::From<(SourceIndex, Token, SourceIndex)> for LexicalToken {
    fn from(triple: (SourceIndex, Token, SourceIndex)) -> LexicalToken {
        LexicalToken(triple.0, triple.1, triple.2)
    }
}

/// Used to identify the type of token expected in a TokenConvertError
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenType {
    Atom,
    Ident,
    String,
    Integer,
    Symbol,
}
impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            TokenType::Atom => write!(f, "ATOM"),
            TokenType::Ident => write!(f, "IDENT"),
            TokenType::String => write!(f, "STRING"),
            TokenType::Integer => write!(f, "INTEGER"),
            TokenType::Symbol => write!(f, "SYMBOL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtomToken(pub SourceIndex, pub Token, pub SourceIndex);
impl AtomToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::Atom(a) => a,
            _ => unreachable!(),
        }
    }
}
impl TryFrom<LexicalToken> for AtomToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<AtomToken> {
        use firefly_intern::symbols;

        match t {
            LexicalToken(start, tok @ Token::Atom(_), end) => {
                return Ok(AtomToken(start, tok, end))
            }
            LexicalToken(start, Token::If, end) => {
                return Ok(AtomToken(start, Token::Atom(symbols::If), end));
            }
            t => Err(TokenConvertError {
                span: t.span(),
                token: t.token(),
                expected: TokenType::Atom,
            }),
        }
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdentToken(pub SourceIndex, pub Token, pub SourceIndex);
impl IdentToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::Ident(a) => a,
            _ => unreachable!(),
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StringToken(pub SourceIndex, pub Token, pub SourceIndex);
impl StringToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
    pub fn symbol(&self) -> Symbol {
        match self.token() {
            Token::String(a) => a,
            _ => unreachable!(),
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IntegerToken(pub SourceIndex, pub Token, pub SourceIndex);
impl IntegerToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
    pub fn small_integer(&self) -> Option<i64> {
        match self.token() {
            Token::Integer(a) => a.to_i64(),
            _ => unreachable!(),
        }
    }
}
impl TryFrom<LexicalToken> for IntegerToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<Self> {
        if let LexicalToken(start, tok @ Token::Integer(_), end) = t {
            return Ok(IntegerToken(start, tok, end));
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::Integer,
        })
    }
}
impl Into<LexicalToken> for IntegerToken {
    fn into(self) -> LexicalToken {
        LexicalToken(self.0, self.1, self.2)
    }
}
impl fmt::Display for IntegerToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolToken(pub SourceIndex, pub Token, pub SourceIndex);
impl SymbolToken {
    pub fn token(&self) -> Token {
        self.1.clone()
    }
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self.0, self.2)
    }
}
impl TryFrom<LexicalToken> for SymbolToken {
    type Error = TokenConvertError;

    fn try_from(t: LexicalToken) -> TokenConvertResult<SymbolToken> {
        match t {
            LexicalToken(_, Token::Atom(_), _) => (),
            LexicalToken(_, Token::Ident(_), _) => (),
            LexicalToken(_, Token::String(_), _) => (),
            LexicalToken(start, token, end) => return Ok(SymbolToken(start, token, end)),
        }
        Err(TokenConvertError {
            span: t.span(),
            token: t.token(),
            expected: TokenType::Symbol,
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DelayedSubstitution {
    Module,
    ModuleString,
    FunctionName,
    FunctionArity,
    File,
    Line,
}
impl fmt::Display for DelayedSubstitution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Module => f.write_str("?MODULE"),
            Self::ModuleString => f.write_str("?MODULE_STRING"),
            Self::FunctionName => f.write_str("?FUNCTION_NAME"),
            Self::FunctionArity => f.write_str("?FUNCTION_ARITY"),
            Self::File => f.write_str("?FILE"),
            Self::Line => f.write_str("?LINE"),
        }
    }
}

/// This enum contains tokens produced by the lexer
#[derive(Debug, Clone)]
pub enum Token {
    // Signifies end of input
    EOF,
    // A tokenization error which may be recovered from
    Error(LexicalError),
    DelayedSubstitution(DelayedSubstitution),
    // Docs
    Comment,
    Edoc,
    // Literals
    Char(char),
    Integer(Int),
    Float(Float),
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
    // Keywords
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
    // Attributes
    Record,
    Spec,
    Callback,
    OptionalCallback,
    Import,
    Export,
    ExportType,
    Removed,
    Module,
    Compile,
    Vsn,
    Author,
    OnLoad,
    Nifs,
    Behaviour,
    Deprecated,
    Type,
    Opaque,
    File,
    // Operators
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
        match self {
            Token::Char(c) => {
                if let Token::Char(c2) = other {
                    return *c == *c2;
                }
            }
            Token::Integer(i) => {
                if let Token::Integer(i2) = other {
                    return *i == *i2;
                }
            }
            Token::Float(n) => {
                if let Token::Float(n2) = other {
                    return *n == *n2;
                }
            }
            Token::Error(_) => {
                if let Token::Error(_) = other {
                    return true;
                }
            }
            Token::Atom(ref a) => {
                if let Token::Atom(a2) = other {
                    return *a == *a2;
                }
            }
            Token::Ident(ref i) => {
                if let Token::Ident(i2) = other {
                    return *i == *i2;
                }
            }
            Token::String(ref s) => {
                if let Token::String(s2) = other {
                    return *s == *s2;
                }
            }
            _ => return mem::discriminant(self) == mem::discriminant(other),
        }
        return false;
    }
}
impl Eq for Token {}
impl Hash for Token {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Token::Float(n) => n.raw().hash(state),
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
            _ => Token::Atom(Symbol::intern(atom)),
        }
    }

    /// For opening tokens like `(` and `[`, get the corresponding
    /// closing token.
    pub fn get_closing_token(&self) -> Self {
        match self {
            Token::LParen => Token::RParen,
            Token::LBrace => Token::RBrace,
            Token::LBracket => Token::RBracket,
            Token::BinaryStart => Token::BinaryEnd,
            _ => panic!("{} has no closing token", self),
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
            Token::DelayedSubstitution(DelayedSubstitution::Module) => write!(f, "ATOM"),
            Token::DelayedSubstitution(DelayedSubstitution::ModuleString) => write!(f, "STRING"),
            Token::DelayedSubstitution(DelayedSubstitution::FunctionName) => write!(f, "ATOM"),
            Token::DelayedSubstitution(DelayedSubstitution::FunctionArity) => write!(f, "INTEGER"),
            Token::DelayedSubstitution(DelayedSubstitution::File) => write!(f, "STRING"),
            Token::DelayedSubstitution(DelayedSubstitution::Line) => write!(f, "INTEGER"),
            // Literals
            Token::Char(ref c) => write!(f, "{}", c),
            Token::Integer(ref i) => write!(f, "{}", i),
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
            Token::Record => write!(f, "record"),
            Token::Spec => write!(f, "spec"),
            Token::Callback => write!(f, "callback"),
            Token::OptionalCallback => write!(f, "optional_callback"),
            Token::Import => write!(f, "import"),
            Token::Export => write!(f, "export"),
            Token::ExportType => write!(f, "export_type"),
            Token::Removed => write!(f, "removed"),
            Token::Module => write!(f, "module"),
            Token::Compile => write!(f, "compile"),
            Token::Vsn => write!(f, "vsn"),
            Token::Author => write!(f, "author"),
            Token::OnLoad => write!(f, "on_load"),
            Token::Nifs => write!(f, "nifs"),
            Token::Behaviour => write!(f, "behaviour"),
            Token::Deprecated => write!(f, "deprecated"),
            Token::Type => write!(f, "type"),
            Token::Opaque => write!(f, "opaque"),
            Token::File => write!(f, "file"),
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
