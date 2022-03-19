//! Guards
//!
//! See: [6.6 Guards](http://erlang.org/doc/apps/erts/absform.html#id88356)
use super::*;

pub type Tuple = common::Tuple<Guard>;
pub type Cons = common::Cons<Guard>;
pub type Binary = common::Binary<Guard>;
pub type UnaryOp = common::UnaryOp<Guard>;
pub type BinaryOp = common::BinaryOp<Guard>;
pub type Record = common::Record<Guard>;
pub type RecordIndex = common::RecordIndex<Guard>;
pub type Map = common::Map<Guard>;
pub type LocalCall = common::LocalCall<Guard>;
pub type RemoteCall = common::RemoteCall<Guard>;

#[derive(Debug, Clone)]
pub struct OrGuard {
    pub and_guards: Vec<Guard>,
}
impl OrGuard {
    pub fn new(and_guards: Vec<Guard>) -> Self {
        OrGuard { and_guards }
    }
}

#[derive(Debug, Clone)]
pub enum Guard {
    Integer(Box<literal::Integer>),
    Float(Box<literal::Float>),
    String(Box<literal::Str>),
    Char(Box<literal::Char>),
    Atom(Box<literal::Atom>),
    Var(Box<common::Var>),
    Tuple(Box<Tuple>),
    Nil(Box<common::Nil>),
    Cons(Box<Cons>),
    Binary(Box<Binary>),
    UnaryOp(Box<UnaryOp>),
    BinaryOp(Box<BinaryOp>),
    Record(Box<Record>),
    RecordIndex(Box<RecordIndex>),
    LocalCall(Box<LocalCall>),
    RemoteCall(Box<RemoteCall>),
}
impl_from!(Guard::Integer(literal::Integer));
impl_from!(Guard::Float(literal::Float));
impl_from!(Guard::String(literal::Str));
impl_from!(Guard::Char(literal::Char));
impl_from!(Guard::Atom(literal::Atom));
impl_from!(Guard::Var(common::Var));
impl_from!(Guard::Tuple(Tuple));
impl_from!(Guard::Nil(common::Nil));
impl_from!(Guard::Cons(Cons));
impl_from!(Guard::Binary(Binary));
impl_from!(Guard::UnaryOp(UnaryOp));
impl_from!(Guard::BinaryOp(BinaryOp));
impl_from!(Guard::Record(Record));
impl_from!(Guard::RecordIndex(RecordIndex));
impl_from!(Guard::LocalCall(LocalCall));
impl_from!(Guard::RemoteCall(RemoteCall));
impl Node for Guard {
    fn line(&self) -> LineNum {
        match *self {
            Guard::Integer(ref x) => x.line(),
            Guard::Float(ref x) => x.line(),
            Guard::String(ref x) => x.line(),
            Guard::Char(ref x) => x.line(),
            Guard::Atom(ref x) => x.line(),
            Guard::Var(ref x) => x.line(),
            Guard::Tuple(ref x) => x.line(),
            Guard::Nil(ref x) => x.line(),
            Guard::Cons(ref x) => x.line(),
            Guard::Binary(ref x) => x.line(),
            Guard::UnaryOp(ref x) => x.line(),
            Guard::BinaryOp(ref x) => x.line(),
            Guard::Record(ref x) => x.line(),
            Guard::RecordIndex(ref x) => x.line(),
            Guard::LocalCall(ref x) => x.line(),
            Guard::RemoteCall(ref x) => x.line(),
        }
    }
}
