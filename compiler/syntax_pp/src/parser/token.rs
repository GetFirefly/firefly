use std::fmt;

use liblumen_diagnostics::SourceIndex;
use liblumen_intern::Symbol;
use liblumen_number::{Float, Integer};
use liblumen_syntax_erl::LexicalError;

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
impl Into<(SourceIndex, Token, SourceIndex)> for LexicalToken {
    fn into(self) -> (SourceIndex, Token, SourceIndex) {
        (self.0, self.1, self.2)
    }
}
impl From<(SourceIndex, Token, SourceIndex)> for LexicalToken {
    fn from(triple: (SourceIndex, Token, SourceIndex)) -> LexicalToken {
        LexicalToken(triple.0, triple.1, triple.2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    EOF,
    Error(LexicalError),

    // Puncutation
    Comma,
    Dot,
    Pipe,
    SquareOpen,
    SquareClose,
    CurlyOpen,
    CurlyClose,

    // Literals
    AtomLiteral(Symbol),
    StringLiteral(Symbol),
    IntegerLiteral(Integer),
    FloatLiteral(Float),

    // Keywords
    Atom,
    Attribute,
    Bin,
    BinElement,
    BitComprehension, // bc
    BitGenerator,     // bc_generate
    Call,
    Callback,
    Case,
    Catch,
    Char,
    Clauses,
    Clause,
    Cons,
    Error,
    Export,
    ExportType,
    File,
    Function,
    Fun,
    If,
    Import,
    Integer,
    ListGenerator,     // generate
    ListComprehension, // lc
    Match,
    Map,
    MapFieldAssoc,
    MapFieldExact,
    Module,
    Nil,
    Op,
    Opaque,
    OptionalCallbacks,
    Receive,
    Record,
    RecordField,
    Spec,
    String,
    Try,
    Tuple,
    Type,
    Var,
    Warning,
}
impl Token {
    pub fn from_bare_atom(atom: &str) -> Self {
        match atom {
            "atom" => Token::Atom,
            "attribute" => Token::Attribute,
            "bin" => Token::Bin,
            "bin_element" => Token::BinElement,
            "bc" => Token::BitComprehension,
            "bc_generate" => Token::BitGenerator,
            "call" => Token::Call,
            "callback" => Token::Callback,
            "case" => Token::Case,
            "catch" => Token::Catch,
            "char" => Token::Char,
            "clauses" => Token::Clauses,
            "clause" => Token::Clause,
            "cons" => Token::Cons,
            "error" => Token::Error,
            "export" => Token::Export,
            "export_type" => Token::ExportType,
            "file" => Token::File,
            "function" => Token::Function,
            "fun" => Token::Fun,
            "if" => Token::If,
            "import" => Token::Import,
            "integer" => Token::Integer,
            "generate" => Token::ListGenerator,
            "lc" => Token::ListComprehension,
            "match" => Token::Match,
            "map" => Token::Map,
            "map_field_assoc" => Token::MapFieldAssoc,
            "map_field_exact" => Token::MapFieldExact,
            "module" => Token::Module,
            "nil" => Token::Nil,
            "op" => Token::Op,
            "opaque" => Token::Opaque,
            "optional_callbacks" => Token::OptionalCallbacks,
            "receive" => Token::Receive,
            "record" => Token::Record,
            "record_field" => Token::RecordField,
            "spec" => Token::Spec,
            "string" => Token::String,
            "try" => Token::Try,
            "tuple" => Token::Tuple,
            "type" => Token::Type,
            "var" => Token::Var,
            "warning" => Token::Warning,
            _ => Token::AtomLiteral(Symbol::intern(atom)),
        }
    }
}
