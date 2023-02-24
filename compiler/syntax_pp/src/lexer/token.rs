use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use firefly_intern::{symbols, Symbol};
use firefly_number::{Float, Int};

use super::LexicalError;

#[derive(Debug, Clone)]
pub enum Token {
    EOF,
    Err(LexicalError),
    Comment,

    // Puncutation
    Bar,
    Comma,
    Dot,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Pound,
    RightArrow,

    // Literals
    CharLiteral(char),
    IntegerLiteral(Int),
    FloatLiteral(Float),
    AtomLiteral(Symbol),
    StringLiteral(Symbol),

    // Keywords
    Any,
    AnnType,
    Atom,
    Attribute,
    Behavior,
    Behaviour,
    Block,
    Bin,
    Binary,
    BinElement,
    BitComprehension, // bc
    BitGenerator,     // bc_generate
    BoundedFun,
    Call,
    Callback,
    Case,
    Catch,
    Char,
    Clause,
    Cons,
    Constraint,
    Compile,
    Default,
    EofAtom,
    Epp,
    Error,
    Export,
    ExportType,
    Float,
    FieldType,
    File,
    Filter,
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
    NamedFun,
    Nifs,
    Nil,
    OnLoad,
    Op,
    Opaque,
    OptionalCallbacks,
    Product,
    Range,
    Receive,
    Record,
    RecordField,
    RecordIndex,
    Remote,
    RemoteType,
    Spec,
    String,
    Try,
    Tuple,
    Type,
    TypedRecordField,
    Union,
    UserType,
    Var,
    Warning,
}
impl Token {
    pub fn from_atom(atom: Symbol) -> Self {
        match atom {
            symbols::Any => Token::Any,
            symbols::AnnType => Token::AnnType,
            symbols::Atom => Token::Atom,
            symbols::Attribute => Token::Attribute,
            symbols::Behavior => Token::Behavior,
            symbols::Behaviour => Token::Behaviour,
            symbols::Bc => Token::BitComprehension,
            symbols::BcGenerate => Token::BitGenerator,
            symbols::Block => Token::Block,
            symbols::Bin => Token::Bin,
            symbols::Binary => Token::Binary,
            symbols::BinElement => Token::BinElement,
            symbols::BoundedFun => Token::BoundedFun,
            symbols::Call => Token::Call,
            symbols::Callback => Token::Callback,
            symbols::Case => Token::Case,
            symbols::Catch => Token::Catch,
            symbols::Char => Token::Char,
            symbols::Clause => Token::Clause,
            symbols::Compile => Token::Compile,
            symbols::Cons => Token::Cons,
            symbols::Constraint => Token::Constraint,
            symbols::Default => Token::Default,
            symbols::Eof => Token::EofAtom,
            symbols::Epp => Token::Epp,
            symbols::Error => Token::Error,
            symbols::Export => Token::Export,
            symbols::ExportType => Token::ExportType,
            symbols::Float => Token::Float,
            symbols::FieldType => Token::FieldType,
            symbols::File => Token::File,
            symbols::Filter => Token::Filter,
            symbols::Function => Token::Function,
            symbols::Fun => Token::Fun,
            symbols::If => Token::If,
            symbols::Import => Token::Import,
            symbols::Integer => Token::Integer,
            symbols::Generate => Token::ListGenerator,
            symbols::Lc => Token::ListComprehension,
            symbols::Match => Token::Match,
            symbols::Map => Token::Map,
            symbols::MapFieldAssoc => Token::MapFieldAssoc,
            symbols::MapFieldExact => Token::MapFieldExact,
            symbols::Module => Token::Module,
            symbols::NamedFun => Token::NamedFun,
            symbols::Nifs => Token::Nifs,
            symbols::Nil => Token::Nil,
            symbols::OnLoad => Token::OnLoad,
            symbols::Op => Token::Op,
            symbols::Opaque => Token::Opaque,
            symbols::OptionalCallbacks => Token::OptionalCallbacks,
            symbols::Product => Token::Product,
            symbols::Range => Token::Range,
            symbols::Receive => Token::Receive,
            symbols::Record => Token::Record,
            symbols::RecordField => Token::RecordField,
            symbols::RecordIndex => Token::RecordIndex,
            symbols::Remote => Token::Remote,
            symbols::RemoteType => Token::RemoteType,
            symbols::Spec => Token::Spec,
            symbols::String => Token::String,
            symbols::Try => Token::Try,
            symbols::Tuple => Token::Tuple,
            symbols::Type => Token::Type,
            symbols::TypedRecordField => Token::TypedRecordField,
            symbols::Union => Token::Union,
            symbols::UserType => Token::UserType,
            symbols::Var => Token::Var,
            symbols::Warning => Token::Warning,
            other => Token::AtomLiteral(other),
        }
    }
}
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        match self {
            Self::EOF => write!(f, "EOF"),
            Self::Err(_) => write!(f, "ERROR"),
            Self::Comment => write!(f, "COMMENT"),
            Self::CharLiteral(ref c) => write!(f, "{}", c),
            Self::IntegerLiteral(ref i) => write!(f, "{}", i),
            Self::FloatLiteral(ref n) => write!(f, "{}", n),
            Self::AtomLiteral(ref s) => write!(f, "'{}'", s),
            Self::StringLiteral(ref s) => write!(f, "\"{}\"", s),

            // Puncutation
            Self::Bar => f.write_char('|'),
            Self::Comma => f.write_char(','),
            Self::Dot => f.write_char('.'),
            Self::LBracket => f.write_char('['),
            Self::RBracket => f.write_char(']'),
            Self::LBrace => f.write_char('{'),
            Self::RBrace => f.write_char('}'),
            Self::Pound => f.write_char('#'),
            Self::RightArrow => write!(f, "=>"),

            // Keywords
            Self::Any => write!(f, "any"),
            Self::AnnType => write!(f, "ann_type"),
            Self::Atom => write!(f, "atom"),
            Self::Attribute => write!(f, "attribute"),
            Self::Behavior => write!(f, "behavior"),
            Self::Behaviour => write!(f, "behaviour"),
            Self::Block => write!(f, "block"),
            Self::Bin => write!(f, "bin"),
            Self::Binary => write!(f, "binary"),
            Self::BinElement => write!(f, "bin_element"),
            Self::BitComprehension => write!(f, "bc"),
            Self::BitGenerator => write!(f, "bc_generate"),
            Self::BoundedFun => write!(f, "bounded_fun"),
            Self::Call => write!(f, "call"),
            Self::Callback => write!(f, "callback"),
            Self::Case => write!(f, "case"),
            Self::Catch => write!(f, "catch"),
            Self::Char => write!(f, "char"),
            Self::Clause => write!(f, "clause"),
            Self::Cons => write!(f, "cons"),
            Self::Constraint => write!(f, "constraint"),
            Self::Compile => write!(f, "compile"),
            Self::Default => write!(f, "default"),
            Self::EofAtom => write!(f, "eof"),
            Self::Epp => write!(f, "epp"),
            Self::Error => write!(f, "error"),
            Self::Export => write!(f, "export"),
            Self::ExportType => write!(f, "export_type"),
            Self::Float => write!(f, "float"),
            Self::FieldType => write!(f, "field_type"),
            Self::File => write!(f, "file"),
            Self::Filter => write!(f, "filter"),
            Self::Function => write!(f, "function"),
            Self::Fun => write!(f, "fun"),
            Self::If => write!(f, "if"),
            Self::Import => write!(f, "import"),
            Self::Integer => write!(f, "integer"),
            Self::ListGenerator => write!(f, "generate"),
            Self::ListComprehension => write!(f, "lc"),
            Self::Match => write!(f, "match"),
            Self::Map => write!(f, "map"),
            Self::MapFieldAssoc => write!(f, "map_field_assoc"),
            Self::MapFieldExact => write!(f, "map_field_exact"),
            Self::Module => write!(f, "module"),
            Self::NamedFun => write!(f, "named_fun"),
            Self::Nifs => write!(f, "nifs"),
            Self::Nil => write!(f, "nil"),
            Self::OnLoad => write!(f, "on_load"),
            Self::Op => write!(f, "op"),
            Self::Opaque => write!(f, "opaque"),
            Self::OptionalCallbacks => write!(f, "optional_callbacks"),
            Self::Product => write!(f, "product"),
            Self::Range => write!(f, "range"),
            Self::Receive => write!(f, "receive"),
            Self::Record => write!(f, "record"),
            Self::RecordField => write!(f, "record_field"),
            Self::RecordIndex => write!(f, "record_index"),
            Self::Remote => write!(f, "remote"),
            Self::RemoteType => write!(f, "remote_type"),
            Self::Spec => write!(f, "spec"),
            Self::String => write!(f, "string"),
            Self::Try => write!(f, "try"),
            Self::Tuple => write!(f, "tuple"),
            Self::Type => write!(f, "type"),
            Self::TypedRecordField => write!(f, "typed_record_field"),
            Self::Union => write!(f, "union"),
            Self::UserType => write!(f, "user_type"),
            Self::Var => write!(f, "var"),
            Self::Warning => write!(f, "warning"),
        }
    }
}
impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        match self {
            Token::CharLiteral(c) => {
                if let Token::CharLiteral(c2) = other {
                    return *c == *c2;
                }
            }
            Token::IntegerLiteral(i) => {
                if let Token::IntegerLiteral(i2) = other {
                    return *i == *i2;
                }
            }
            Token::FloatLiteral(n) => {
                if let Token::FloatLiteral(n2) = other {
                    return *n == *n2;
                }
            }
            Token::Err(_) => {
                if let Token::Err(_) = other {
                    return true;
                }
            }
            Token::AtomLiteral(ref a) => {
                if let Token::AtomLiteral(a2) = other {
                    return *a == *a2;
                }
            }
            Token::StringLiteral(ref s) => {
                if let Token::StringLiteral(s2) = other {
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
            Token::FloatLiteral(n) => n.raw().hash(state),
            Token::Err(ref e) => e.hash(state),
            Token::AtomLiteral(ref a) => a.hash(state),
            Token::StringLiteral(ref s) => s.hash(state),
            Token::CharLiteral(c) => c.hash(state),
            ref token => token.to_string().hash(state),
        }
    }
}
