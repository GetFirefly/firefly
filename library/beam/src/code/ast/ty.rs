//! Types
//!
//! See: [6.7 Types](http://erlang.org/doc/apps/erts/absform.html#id88630)
use firefly_intern::Symbol;

use super::*;

#[derive(Debug, Clone)]
pub enum Type {
    Any(AnyType),
    Atom(Atom),
    Integer(Box<Integer>),
    Var(Var),
    Annotated(Box<AnnotatedType>),
    UnaryOp(Box<UnaryTypeOp>),
    BinaryOp(Box<BinaryTypeOp>),
    BitString(Box<BitStringType>),
    Nil(Nil),
    AnyFun(Box<AnyFunType>),
    Function(Box<FunType>),
    Range(Box<RangeType>),
    Map(Box<MapType>),
    BuiltIn(Box<BuiltInType>),
    Record(Box<RecordType>),
    Remote(Box<RemoteType>),
    AnyTuple(Box<AnyTupleType>),
    Tuple(Box<TupleType>),
    Union(Box<UnionType>),
    Product(Box<ProductType>),
    User(Box<UserType>),
}
impl_from!(Type::Any(AnyType));
impl_from!(Type::Atom(Atom));
impl_from!(Type::Integer(Integer));
impl_from!(Type::Var(Var));
impl_from!(Type::Annotated(AnnotatedType));
impl_from!(Type::UnaryOp(UnaryTypeOp));
impl_from!(Type::BinaryOp(BinaryTypeOp));
impl_from!(Type::BitString(BitStringType));
impl_from!(Type::Nil(Nil));
impl_from!(Type::AnyFun(AnyFunType));
impl_from!(Type::Function(FunType));
impl_from!(Type::Range(RangeType));
impl_from!(Type::Map(MapType));
impl_from!(Type::BuiltIn(BuiltInType));
impl_from!(Type::Record(RecordType));
impl_from!(Type::Remote(RemoteType));
impl_from!(Type::AnyTuple(AnyTupleType));
impl_from!(Type::Tuple(TupleType));
impl_from!(Type::Union(UnionType));
impl_from!(Type::Product(ProductType));
impl_from!(Type::User(UserType));
impl Node for Type {
    fn loc(&self) -> Location {
        match self {
            Self::Any(ref x) => x.loc(),
            Self::Atom(ref x) => x.loc(),
            Self::Integer(ref x) => x.loc(),
            Self::Var(ref x) => x.loc(),
            Self::Annotated(ref x) => x.loc(),
            Self::UnaryOp(ref x) => x.loc(),
            Self::BinaryOp(ref x) => x.loc(),
            Self::BitString(ref x) => x.loc(),
            Self::Nil(ref x) => x.loc(),
            Self::AnyFun(ref x) => x.loc(),
            Self::Function(ref x) => x.loc(),
            Self::Range(ref x) => x.loc(),
            Self::Map(ref x) => x.loc(),
            Self::BuiltIn(ref x) => x.loc(),
            Self::Record(ref x) => x.loc(),
            Self::Remote(ref x) => x.loc(),
            Self::AnyTuple(ref x) => x.loc(),
            Self::Tuple(ref x) => x.loc(),
            Self::Union(ref x) => x.loc(),
            Self::Product(ref x) => x.loc(),
            Self::User(ref x) => x.loc(),
        }
    }
}
impl Type {
    pub fn any(loc: Location) -> Self {
        Self::Any(AnyType::new(loc))
    }
}

#[derive(Debug, Clone)]
pub struct AnyType {
    pub loc: Location,
}
impl_node!(AnyType);
impl AnyType {
    pub fn new(loc: Location) -> Self {
        Self { loc }
    }
}

#[derive(Debug, Clone)]
pub struct UnaryTypeOp {
    pub loc: Location,
    pub operator: Symbol,
    pub operand: Expression,
}
impl_node!(UnaryTypeOp);
impl UnaryTypeOp {
    pub fn new(loc: Location, operator: Symbol, operand: Expression) -> Self {
        Self {
            loc,
            operator,
            operand,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTypeOp {
    pub loc: Location,
    pub operator: Symbol,
    pub left_operand: Expression,
    pub right_operand: Expression,
}
impl_node!(BinaryTypeOp);
impl BinaryTypeOp {
    pub fn new(
        loc: Location,
        operator: Symbol,
        left_operand: Expression,
        right_operand: Expression,
    ) -> Self {
        Self {
            loc,
            operator,
            left_operand,
            right_operand,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UserType {
    pub loc: Location,
    pub name: Symbol,
    pub args: Vec<Type>,
}
impl_node!(UserType);
impl UserType {
    pub fn new(loc: Location, name: Symbol, args: Vec<Type>) -> Self {
        Self { loc, name, args }
    }
}

#[derive(Debug, Clone)]
pub struct UnionType {
    pub loc: Location,
    pub types: Vec<Type>,
}
impl_node!(UnionType);
impl UnionType {
    pub fn new(loc: Location, types: Vec<Type>) -> Self {
        Self { loc, types }
    }
}

#[derive(Debug, Clone)]
pub struct ProductType {
    pub loc: Location,
    pub types: Vec<Type>,
}
impl_node!(ProductType);
impl ProductType {
    pub fn new(loc: Location, types: Vec<Type>) -> Self {
        Self { loc, types }
    }
}

#[derive(Debug, Clone)]
pub struct AnyTupleType {
    pub loc: Location,
}
impl_node!(AnyTupleType);
impl AnyTupleType {
    pub fn new(loc: Location) -> Self {
        Self { loc }
    }
}

#[derive(Debug, Clone)]
pub struct TupleType {
    pub loc: Location,
    pub elements: Vec<Type>,
}
impl_node!(TupleType);
impl TupleType {
    pub fn new(loc: Location, elements: Vec<Type>) -> Self {
        Self { loc, elements }
    }
}

#[derive(Debug, Clone)]
pub struct RemoteType {
    pub loc: Location,
    pub name: FunctionName,
    pub args: Vec<Type>,
}
impl_node!(RemoteType);
impl RemoteType {
    pub fn new(loc: Location, name: FunctionName, args: Vec<Type>) -> Self {
        Self { loc, name, args }
    }
}

#[derive(Debug, Clone)]
pub struct RecordType {
    pub loc: Location,
    pub name: Symbol,
    pub fields: Vec<RecordFieldType>,
}
impl_node!(RecordType);
impl RecordType {
    pub fn new(loc: Location, name: Symbol, fields: Vec<RecordFieldType>) -> Self {
        Self { loc, name, fields }
    }
}

#[derive(Debug, Clone)]
pub struct RecordFieldType {
    pub loc: Location,
    pub name: Symbol,
    pub ty: Type,
}
impl_node!(RecordFieldType);
impl RecordFieldType {
    pub fn new(loc: Location, name: Symbol, ty: Type) -> Self {
        Self { loc, name, ty }
    }
}

#[derive(Debug, Clone)]
pub struct BuiltInType {
    pub loc: Location,
    pub name: Symbol,
    pub args: Vec<Type>,
}
impl_node!(BuiltInType);
impl BuiltInType {
    pub fn new(loc: Location, name: Symbol, args: Vec<Type>) -> Self {
        Self { loc, name, args }
    }
}

#[derive(Debug, Clone)]
pub struct MapType {
    pub loc: Location,
    pub pairs: Vec<MapPairType>,
}
impl_node!(MapType);
impl MapType {
    pub fn new(loc: Location, pairs: Vec<MapPairType>) -> Self {
        Self { loc, pairs }
    }
}

#[derive(Debug, Clone)]
pub struct MapPairType {
    pub loc: Location,
    pub key: Type,
    pub value: Type,
}
impl_node!(MapPairType);
impl MapPairType {
    pub fn new(loc: Location, key: Type, value: Type) -> Self {
        Self { loc, key, value }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotatedType {
    pub loc: Location,
    pub name: expr::Var,
    pub ty: Type,
}
impl_node!(AnnotatedType);
impl AnnotatedType {
    pub fn new(loc: Location, name: expr::Var, ty: Type) -> Self {
        Self { loc, name, ty }
    }
}

#[derive(Debug, Clone)]
pub struct BitStringType {
    pub loc: Location,
    pub size: u64,
    pub tail_bits: u64,
}
impl_node!(BitStringType);
impl BitStringType {
    pub fn new(loc: Location, size: u64, tail_bits: u64) -> Self {
        Self {
            loc,
            size,
            tail_bits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnyFunType {
    pub loc: Location,
    pub return_type: Option<Type>,
}
impl_node!(AnyFunType);
impl AnyFunType {
    pub fn new(loc: Location, return_type: Option<Type>) -> Self {
        Self { loc, return_type }
    }
}

#[derive(Debug, Clone)]
pub struct FunType {
    pub loc: Location,
    pub args: Vec<Type>,
    pub return_type: Type,
    pub constraints: Vec<Constraint>,
}
impl_node!(FunType);
impl FunType {
    pub fn new(loc: Location, args: Vec<Type>, return_type: Type) -> Self {
        Self {
            loc,
            args,
            return_type,
            constraints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub loc: Location,
    pub var: expr::Var,
    pub subtype: Type,
}
impl_node!(Constraint);
impl Constraint {
    pub fn new(loc: Location, var: expr::Var, subtype: Type) -> Self {
        Self { loc, var, subtype }
    }
}

#[derive(Debug, Clone)]
pub struct RangeType {
    pub loc: Location,
    pub low: Type,
    pub high: Type,
}
impl_node!(RangeType);
impl RangeType {
    pub fn new(loc: Location, low: Type, high: Type) -> Self {
        Self { loc, low, high }
    }
}
