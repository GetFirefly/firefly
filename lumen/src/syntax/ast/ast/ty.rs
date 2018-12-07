//! Types
//!
//! See: [6.7 Types](http://erlang.org/doc/apps/erts/absform.html#id88630)
use super::common;
use super::literal;
use super::{LineNum, Node};

pub type UnaryOp = common::UnaryOp<Type>;
pub type BinaryOp = common::BinaryOp<Type>;

#[derive(Debug, Clone)]
pub enum Type {
    Atom(Box<literal::Atom>),
    Integer(Box<literal::Integer>),
    Var(Box<common::Var>),
    Annotated(Box<Annotated>),
    UnaryOp(Box<UnaryOp>),
    BinaryOp(Box<BinaryOp>),
    BitString(Box<BitString>),
    Nil(Box<common::Nil>),
    AnyFun(Box<AnyFun>),
    Function(Box<Fun>),
    Range(Box<Range>),
    Map(Box<Map>),
    BuiltIn(Box<BuiltInType>),
    Record(Box<Record>),
    Remote(Box<RemoteType>),
    AnyTuple(Box<AnyTuple>),
    Tuple(Box<Tuple>),
    Union(Box<Union>),
    User(Box<UserType>),
}
impl_from!(Type::Atom(literal::Atom));
impl_from!(Type::Integer(literal::Integer));
impl_from!(Type::Var(common::Var));
impl_from!(Type::Annotated(Annotated));
impl_from!(Type::UnaryOp(UnaryOp));
impl_from!(Type::BinaryOp(BinaryOp));
impl_from!(Type::BitString(BitString));
impl_from!(Type::Nil(common::Nil));
impl_from!(Type::AnyFun(AnyFun));
impl_from!(Type::Function(Fun));
impl_from!(Type::Range(Range));
impl_from!(Type::Map(Map));
impl_from!(Type::BuiltIn(BuiltInType));
impl_from!(Type::Record(Record));
impl_from!(Type::Remote(RemoteType));
impl_from!(Type::AnyTuple(AnyTuple));
impl_from!(Type::Tuple(Tuple));
impl_from!(Type::Union(Union));
impl_from!(Type::User(UserType));
impl Node for Type {
    fn line(&self) -> LineNum {
        match *self {
            Type::Integer(ref x) => x.line(),
            Type::Atom(ref x) => x.line(),
            Type::Var(ref x) => x.line(),
            Type::Annotated(ref x) => x.line(),
            Type::UnaryOp(ref x) => x.line(),
            Type::BinaryOp(ref x) => x.line(),
            Type::BitString(ref x) => x.line(),
            Type::Nil(ref x) => x.line(),
            Type::AnyFun(ref x) => x.line(),
            Type::Function(ref x) => x.line(),
            Type::Range(ref x) => x.line(),
            Type::Map(ref x) => x.line(),
            Type::BuiltIn(ref x) => x.line(),
            Type::Record(ref x) => x.line(),
            Type::Remote(ref x) => x.line(),
            Type::AnyTuple(ref x) => x.line(),
            Type::Tuple(ref x) => x.line(),
            Type::Union(ref x) => x.line(),
            Type::User(ref x) => x.line(),
        }
    }
}
impl Type {
    pub fn any(line: LineNum) -> Self {
        Type::BuiltIn(Box::new(BuiltInType::new(
            line,
            "any".to_string(),
            Vec::new(),
        )))
    }
}

#[derive(Debug, Clone)]
pub struct UserType {
    pub line: LineNum,
    pub name: String,
    pub args: Vec<Type>,
}
impl_node!(UserType);
impl UserType {
    pub fn new(line: LineNum, name: String, args: Vec<Type>) -> Self {
        UserType {
            line: line,
            name: name,
            args: args,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Union {
    pub line: LineNum,
    pub types: Vec<Type>,
}
impl_node!(Union);
impl Union {
    pub fn new(line: LineNum, types: Vec<Type>) -> Self {
        Union {
            line: line,
            types: types,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnyTuple {
    pub line: LineNum,
}
impl_node!(AnyTuple);
impl AnyTuple {
    pub fn new(line: LineNum) -> Self {
        AnyTuple { line: line }
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub line: LineNum,
    pub elements: Vec<Type>,
}
impl_node!(Tuple);
impl Tuple {
    pub fn new(line: LineNum, elements: Vec<Type>) -> Self {
        Tuple {
            line: line,
            elements: elements,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RemoteType {
    pub line: LineNum,
    pub module: String,
    pub function: String,
    pub args: Vec<Type>,
}
impl_node!(RemoteType);
impl RemoteType {
    pub fn new(line: LineNum, module: String, function: String, args: Vec<Type>) -> Self {
        RemoteType {
            line: line,
            module: module,
            function: function,
            args: args,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub line: LineNum,
    pub name: String,
    pub fields: Vec<RecordField>,
}
impl_node!(Record);
impl Record {
    pub fn new(line: LineNum, name: String, fields: Vec<RecordField>) -> Self {
        Record {
            line: line,
            name: name,
            fields: fields,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordField {
    pub line: LineNum,
    pub name: String,
    pub ty: Type,
}
impl_node!(RecordField);
impl RecordField {
    pub fn new(line: LineNum, name: String, ty: Type) -> Self {
        RecordField {
            line: line,
            name: name,
            ty: ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuiltInType {
    pub line: LineNum,
    pub name: String,
    pub args: Vec<Type>,
}
impl_node!(BuiltInType);
impl BuiltInType {
    pub fn new(line: LineNum, name: String, args: Vec<Type>) -> Self {
        BuiltInType {
            line: line,
            name: name,
            args: args,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Map {
    pub line: LineNum,
    pub pairs: Vec<MapPair>,
}
impl_node!(Map);
impl Map {
    pub fn new(line: LineNum, pairs: Vec<MapPair>) -> Self {
        Map {
            line: line,
            pairs: pairs,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapPair {
    pub line: LineNum,
    pub key: Type,
    pub value: Type,
}
impl_node!(MapPair);
impl MapPair {
    pub fn new(line: LineNum, key: Type, value: Type) -> Self {
        MapPair {
            line: line,
            key: key,
            value: value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Annotated {
    pub line: LineNum,
    pub name: common::Var,
    pub ty: Type,
}
impl_node!(Annotated);
impl Annotated {
    pub fn new(line: LineNum, name: common::Var, ty: Type) -> Self {
        Annotated {
            line: line,
            name: name,
            ty: ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitString {
    pub line: LineNum,
    pub bytes: u64,
    pub tail_bits: u64,
}
impl_node!(BitString);
impl BitString {
    pub fn new(line: LineNum, bytes: u64, tail_bits: u64) -> Self {
        BitString {
            line: line,
            bytes: bytes,
            tail_bits: tail_bits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnyFun {
    pub line: LineNum,
    pub return_type: Option<Type>,
}
impl_node!(AnyFun);
impl AnyFun {
    pub fn new(line: LineNum) -> Self {
        AnyFun {
            line: line,
            return_type: None,
        }
    }
    pub fn return_type(mut self, return_type: Type) -> Self {
        self.return_type = Some(return_type);
        self
    }
}

#[derive(Debug, Clone)]
pub struct Fun {
    pub line: LineNum,
    pub args: Vec<Type>,
    pub return_type: Type,
    pub constraints: Vec<Constraint>,
}
impl_node!(Fun);
impl Fun {
    pub fn new(line: LineNum, args: Vec<Type>, return_type: Type) -> Self {
        Fun {
            line: line,
            args: args,
            return_type: return_type,
            constraints: Vec::new(),
        }
    }
    pub fn constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub line: LineNum,
    pub var: common::Var,
    pub subtype: Type,
}
impl_node!(Constraint);
impl Constraint {
    pub fn new(line: LineNum, var: common::Var, subtype: Type) -> Self {
        Constraint {
            line: line,
            var: var,
            subtype: subtype,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Range {
    pub line: LineNum,
    pub low: Type,
    pub high: Type,
}
impl_node!(Range);
impl Range {
    pub fn new(line: LineNum, low: Type, high: Type) -> Self {
        Range {
            line: line,
            low: low,
            high: high,
        }
    }
}
