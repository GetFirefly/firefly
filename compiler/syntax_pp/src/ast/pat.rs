//! Patterns
//!
//! See: [6.3 Patterns](http://erlang.org/doc/apps/erts/absform.html#id87135)
use super::common;
use super::literal;
use super::{LineNum, Node};

pub type Match = common::Match<Pattern, Pattern>;
pub type Tuple = common::Tuple<Pattern>;
pub type Cons = common::Cons<Pattern>;
pub type Binary = common::Binary<Pattern>;
pub type UnaryOp = common::UnaryOp<Pattern>;
pub type BinaryOp = common::BinaryOp<Pattern>;
pub type Record = common::Record<Pattern>;
pub type RecordIndex = common::RecordIndex<Pattern>;
pub type Map = common::Map<Pattern>;

#[derive(Debug, Clone)]
pub enum Pattern {
    Integer(Box<literal::Integer>),
    Float(Box<literal::Float>),
    String(Box<literal::Str>),
    Char(Box<literal::Char>),
    Atom(Box<literal::Atom>),
    Var(Box<common::Var>),
    Match(Box<Match>),
    Tuple(Box<Tuple>),
    Nil(Box<common::Nil>),
    Cons(Box<Cons>),
    Binary(Box<Binary>),
    UnaryOp(Box<UnaryOp>),
    BinaryOp(Box<BinaryOp>),
    Record(Box<Record>),
    RecordIndex(Box<RecordIndex>),
    Map(Box<Map>),
}
impl_from!(Pattern::Integer(literal::Integer));
impl_from!(Pattern::Float(literal::Float));
impl_from!(Pattern::String(literal::Str));
impl_from!(Pattern::Char(literal::Char));
impl_from!(Pattern::Atom(literal::Atom));
impl_from!(Pattern::Var(common::Var));
impl_from!(Pattern::Match(Match));
impl_from!(Pattern::Tuple(Tuple));
impl_from!(Pattern::Nil(common::Nil));
impl_from!(Pattern::Cons(Cons));
impl_from!(Pattern::Binary(Binary));
impl_from!(Pattern::UnaryOp(UnaryOp));
impl_from!(Pattern::BinaryOp(BinaryOp));
impl_from!(Pattern::Record(Record));
impl_from!(Pattern::RecordIndex(RecordIndex));
impl_from!(Pattern::Map(Map));
impl Node for Pattern {
    fn line(&self) -> LineNum {
        match *self {
            Pattern::Integer(ref x) => x.line(),
            Pattern::Float(ref x) => x.line(),
            Pattern::String(ref x) => x.line(),
            Pattern::Char(ref x) => x.line(),
            Pattern::Atom(ref x) => x.line(),
            Pattern::Var(ref x) => x.line(),
            Pattern::Match(ref x) => x.line(),
            Pattern::Tuple(ref x) => x.line(),
            Pattern::Nil(ref x) => x.line(),
            Pattern::Cons(ref x) => x.line(),
            Pattern::Binary(ref x) => x.line(),
            Pattern::UnaryOp(ref x) => x.line(),
            Pattern::BinaryOp(ref x) => x.line(),
            Pattern::Record(ref x) => x.line(),
            Pattern::RecordIndex(ref x) => x.line(),
            Pattern::Map(ref x) => x.line(),
        }
    }
}
