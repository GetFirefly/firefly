use std::cmp::Ordering;

use rug::Integer;
use liblumen_diagnostics::ByteSpan;

use super::{Ident, BinaryOp, UnaryOp};
use super::{Type, Guard, Name, FunctionName, Function};

/// The set of all possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // An identifier/variable/function reference
    Var(Ident),
    Literal(Literal),
    FunctionName(FunctionName),
    // The various list forms
    Nil(Nil),
    Cons(Cons),
    // Other data structures
    Tuple(Tuple),
    Map(Map),
    MapUpdate(MapUpdate),
    MapProjection(MapProjection),
    Binary(Binary),
    Record(Record),
    RecordAccess(RecordAccess),
    RecordIndex(RecordIndex),
    RecordUpdate(RecordUpdate),
    // Comprehensions
    ListComprehension(ListComprehension),
    BinaryComprehension(BinaryComprehension),
    Generator(Generator),
    BinaryGenerator(BinaryGenerator),
    // Complex expressions
    Begin(Begin),
    Apply(Apply),
    Remote(Remote),
    BinaryExpr(BinaryExpr),
    UnaryExpr(UnaryExpr),
    Match(Match),
    If(If),
    Catch(Catch),
    Case(Case),
    Receive(Receive),
    Try(Try),
    Fun(Function),
}
impl Expr {
    pub fn span(&self) -> ByteSpan {
        match self {
            &Expr::Var(Ident { ref span, .. }) => span.clone(),
            &Expr::Literal(ref lit) => lit.span(),
            &Expr::FunctionName(ref name) => name.span(),
            &Expr::Nil(Nil(ref span)) => span.clone(),
            &Expr::Cons(Cons { ref span, .. }) => span.clone(),
            &Expr::Tuple(Tuple { ref span, .. }) => span.clone(),
            &Expr::Map(Map { ref span, .. }) => span.clone(),
            &Expr::MapUpdate(MapUpdate { ref span, .. }) => span.clone(),
            &Expr::MapProjection(MapProjection { ref span, .. }) => span.clone(),
            &Expr::Binary(Binary { ref span, .. }) => span.clone(),
            &Expr::Record(Record { ref span, .. }) => span.clone(),
            &Expr::RecordAccess(RecordAccess { ref span, .. }) => span.clone(),
            &Expr::RecordIndex(RecordIndex { ref span, .. }) => span.clone(),
            &Expr::RecordUpdate(RecordUpdate { ref span, .. }) => span.clone(),
            &Expr::ListComprehension(ListComprehension { ref span, .. }) => span.clone(),
            &Expr::BinaryComprehension(BinaryComprehension { ref span, .. }) => span.clone(),
            &Expr::Generator(Generator { ref span, .. }) => span.clone(),
            &Expr::BinaryGenerator(BinaryGenerator { ref span, .. }) => span.clone(),
            &Expr::Begin(Begin { ref span, .. }) => span.clone(),
            &Expr::Apply(Apply { ref span, .. }) => span.clone(),
            &Expr::Remote(Remote { ref span, .. }) => span.clone(),
            &Expr::BinaryExpr(BinaryExpr { ref span, .. }) => span.clone(),
            &Expr::UnaryExpr(UnaryExpr { ref span, .. }) => span.clone(),
            &Expr::Match(Match { ref span, .. }) => span.clone(),
            &Expr::If(If { ref span, .. }) => span.clone(),
            &Expr::Catch(Catch { ref span, .. }) => span.clone(),
            &Expr::Case(Case { ref span, .. }) => span.clone(),
            &Expr::Receive(Receive { ref span, .. }) => span.clone(),
            &Expr::Try(Try { ref span, .. }) => span.clone(),
            &Expr::Fun(ref fun) => fun.span(),
        }
    }
}
impl PartialOrd for Expr {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn partial_cmp(&self, other: &Expr) -> Option<Ordering> {
        match (self, other) {
            (&Expr::Binary(_), &Expr::Binary(_)) => None,
            (&Expr::Binary(_), _) => Some(Ordering::Greater),
            (_, &Expr::Binary(_)) => Some(Ordering::Less),
            (&Expr::Cons(ref lhs), &Expr::Cons(ref rhs)) => lhs.partial_cmp(rhs),
            (&Expr::Cons(_), _) => Some(Ordering::Greater),
            (_, &Expr::Cons(_)) => Some(Ordering::Less),
            (&Expr::Nil(_), &Expr::Nil(_)) => Some(Ordering::Equal),
            (&Expr::Nil(_), _) => Some(Ordering::Greater),
            (_, &Expr::Nil(_)) => Some(Ordering::Less),
            (&Expr::Map(ref lhs), &Expr::Map(ref rhs)) => lhs.partial_cmp(rhs),
            (&Expr::Map(_), _) => Some(Ordering::Greater),
            (_, &Expr::Map(_)) => Some(Ordering::Less),
            (&Expr::Tuple(ref lhs), &Expr::Tuple(ref rhs)) => lhs.partial_cmp(rhs),
            (&Expr::Tuple(_), _) => Some(Ordering::Greater),
            (_, &Expr::Tuple(_)) => Some(Ordering::Less),
            (&Expr::Fun(_), &Expr::Fun(_)) => None,
            (&Expr::Fun(_), _) => Some(Ordering::Greater),
            (_, &Expr::Fun(_)) => Some(Ordering::Less),
            (&Expr::Literal(ref lhs), &Expr::Literal(ref rhs)) => lhs.partial_cmp(rhs),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Nil(pub ByteSpan);
impl PartialEq for Nil {
    fn eq(&self, _: &Self) -> bool {
        return true;
    }
}
impl Eq for Nil {}

#[derive(Debug, Clone)]
pub struct Cons {
    pub span: ByteSpan,
    pub head: Box<Expr>,
    pub tail: Box<Expr>,
}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}
impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.head.partial_cmp(&other.head) {
            None => self.tail.partial_cmp(&other.tail),
            Some(order) => Some(order),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub span: ByteSpan,
    pub elements: Vec<Expr>,
}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}
impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.elements.partial_cmp(&other.elements)
    }
}

#[derive(Debug, Clone)]
pub struct Map {
    pub span: ByteSpan,
    pub fields: Vec<MapField>,
}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl PartialOrd for Map {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fields.partial_cmp(&other.fields)
    }
}

// Updating fields on an existing map, e.g. `Map#{field1 = value1}.`
#[derive(Debug, Clone)]
pub struct MapUpdate {
    pub span: ByteSpan,
    pub map: Box<Expr>,
    pub updates: Vec<MapField>,
}
impl PartialEq for MapUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map && self.updates == other.updates
    }
}

// Pattern matching a map expression
#[derive(Debug, Clone)]
pub struct MapProjection {
    pub span: ByteSpan,
    pub map: Box<Expr>,
    pub fields: Vec<MapField>,
}
impl PartialEq for MapProjection {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map && self.fields == other.fields
    }
}

/// The set of literal values
///
/// This does not include tuples, lists, and maps,
/// even though those can be constructed at compile-time,
/// as some places that allow literals do not permit those
/// types
#[derive(Debug, Clone)]
pub enum Literal {
    Atom(Ident),
    String(Ident),
    Char(ByteSpan, char),
    Integer(ByteSpan, i64),
    BigInteger(ByteSpan, Integer),
    Float(ByteSpan, f64),
}
impl Literal {
    pub fn span(&self) -> ByteSpan {
        match self {
            &Literal::Atom(Ident { ref span, .. }) => span.clone(),
            &Literal::String(Ident { ref span, .. }) => span.clone(),
            &Literal::Char(ref span, _) => span.clone(),
            &Literal::Integer(ref span, _) => span.clone(),
            &Literal::BigInteger(ref span, _) => span.clone(),
            &Literal::Float(ref span, _) => span.clone(),
        }
    }
}
impl PartialEq for Literal {
    fn eq(&self, other: &Literal) -> bool {
        match (self, other) {
            (&Literal::Atom(ref lhs), &Literal::Atom(ref rhs)) =>
                lhs == rhs,
            (&Literal::Atom(_), _) => false,
            (_, &Literal::Atom(_)) => false,
            (&Literal::String(ref lhs), &Literal::String(ref rhs)) =>
                lhs == rhs,
            (&Literal::String(_), _) => false,
            (_, &Literal::String(_)) => false,
            (x, y) => x.partial_cmp(y) == Some(Ordering::Equal),
        }
    }
}
impl PartialOrd for Literal {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn partial_cmp(&self, other: &Literal) -> Option<Ordering> {
        match (self, other) {
            (&Literal::String(ref lhs), &Literal::String(ref rhs)) =>
                lhs.partial_cmp(rhs),
            (&Literal::String(_), _) => Some(Ordering::Greater),
            (_, &Literal::String(_)) => Some(Ordering::Less),
            (&Literal::Atom(ref lhs), &Literal::Atom(ref rhs)) =>
                lhs.partial_cmp(rhs),
            (&Literal::Atom(_), _) => Some(Ordering::Greater),
            (_, &Literal::Atom(_)) => Some(Ordering::Less),
            (&Literal::Integer(_, x), &Literal::Integer(_, y)) => x.partial_cmp(&y),
            (&Literal::Integer(_, x), &Literal::BigInteger(_, ref y)) => x.partial_cmp(y),
            (&Literal::Integer(_, x), &Literal::Float(_, y)) => (x as f64).partial_cmp(&y),
            (&Literal::Integer(_, x), &Literal::Char(_, y)) => x.partial_cmp(&(y as i64)),
            (&Literal::BigInteger(_, ref x), &Literal::BigInteger(_, ref y)) => x.partial_cmp(y),
            (&Literal::BigInteger(_, ref x), &Literal::Integer(_, y)) => x.partial_cmp(&y),
            (&Literal::BigInteger(_, ref x), &Literal::Float(_, y)) => x.partial_cmp(&y),
            (&Literal::BigInteger(_, ref x), &Literal::Char(_, y)) => x.partial_cmp(&(y as i64)),
            (&Literal::Float(_, x), &Literal::Float(_, y)) => x.partial_cmp(&y),
            (&Literal::Float(_, x), &Literal::Integer(_, y)) => x.partial_cmp(&(y as f64)),
            (&Literal::Float(_, x), &Literal::BigInteger(_, ref y)) => x.partial_cmp(y),
            (&Literal::Float(_, x), &Literal::Char(_, y)) => x.partial_cmp(&((y as i64) as f64)),
            (&Literal::Char(_, x), &Literal::Char(_, y)) => x.partial_cmp(&y),
            (&Literal::Char(_, x), &Literal::Integer(_, y)) => (x as i64).partial_cmp(&y),
            (&Literal::Char(_, x), &Literal::BigInteger(_, ref y)) => (x as i64).partial_cmp(y),
            (&Literal::Char(_, x), &Literal::Float(_, y)) => ((x as i64) as f64).partial_cmp(&y),
        }
    }
}

/// Maps can have two different types of field assignment:
///
/// * assoc - inserts or updates the given key with the given value
/// * exact - updates the given key with the given value, or produces an error
#[derive(Debug, Clone)]
pub enum MapField {
    Assoc {
        span: ByteSpan,
        key: Expr,
        value: Expr,
    },
    Exact {
        span: ByteSpan,
        key: Expr,
        value: Expr,
    },
}
impl MapField {
    pub fn key(&self) -> Expr {
        match self {
            &MapField::Assoc { ref key, .. } => key.clone(),
            &MapField::Exact { ref key, .. } => key.clone(),
        }
    }

    pub fn value(&self) -> Expr {
        match self {
            &MapField::Assoc { ref value, .. } => value.clone(),
            &MapField::Exact { ref value, .. } => value.clone(),
        }
    }
}
impl PartialEq for MapField {
    fn eq(&self, other: &Self) -> bool {
        (self.key() == other.key()) && (self.value() == other.value())
    }
}
impl PartialOrd for MapField {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.key().partial_cmp(&other.key()) {
            None => None,
            Some(Ordering::Equal) => self.value().partial_cmp(&other.value()),
            Some(order) => Some(order),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub span: ByteSpan,
    pub name: Ident,
    pub fields: Vec<RecordField>,
}
impl PartialEq for Record {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.fields == other.fields
    }
}

// Accessing a record field value, e.g. Expr#myrec.field1
#[derive(Debug, Clone)]
pub struct RecordAccess {
    pub span: ByteSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordAccess {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name &&
        self.field == other.field &&
        self.record == other.record
    }
}

// Referencing a record fields index, e.g. #myrec.field1
#[derive(Debug, Clone)]
pub struct RecordIndex {
    pub span: ByteSpan,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordIndex {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.field == other.field
    }
}

// Update a record field value, e.g. Expr#myrec.field1
#[derive(Debug, Clone)]
pub struct RecordUpdate {
    pub span: ByteSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub updates: Vec<RecordField>,
}
impl PartialEq for RecordUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name &&
        self.record == other.record &&
        self.updates == other.updates
    }
}

/// Record fields always have a name, but both default value and type
/// are optional in a record definition. When instantiating a record,
/// if no value is given for a field, and no default is given,
/// then `undefined` is the default.
#[derive(Debug, Clone)]
pub struct RecordField {
    pub span: ByteSpan,
    pub name: Name,
    pub value: Option<Expr>,
    pub ty: Option<Type>,
}
impl PartialEq for RecordField {
    fn eq(&self, other: &Self) -> bool {
        (self.name == other.name) && (self.value == other.value) && (self.ty == other.ty)
    }
}

#[derive(Debug, Clone)]
pub struct Binary {
    pub span: ByteSpan,
    pub elements: Vec<BinaryElement>,
}
impl PartialEq for Binary {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

/// Used to represent a specific segment in a binary constructor, to
/// produce a binary, all segments must be evaluated, and then assembled
#[derive(Debug, Clone)]
pub struct BinaryElement {
    pub span: ByteSpan,
    pub bit_expr: Expr,
    pub bit_size: Option<Expr>,
    pub bit_type: Option<Vec<BitType>>,
}
impl PartialEq for BinaryElement {
    fn eq(&self, other: &Self) -> bool {
        (self.bit_expr == other.bit_expr)
            && (self.bit_size == other.bit_size)
            && (self.bit_type == other.bit_type)
    }
}

/// A bit type can come in the form `Type` or `Type:Size`
#[derive(Debug, Clone)]
pub enum BitType {
    Name(ByteSpan, Ident),
    Sized(ByteSpan, Ident, i64),
}
impl PartialEq for BitType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&BitType::Name(_, ref x1), &BitType::Name(_, ref y1)) => x1 == y1,
            (&BitType::Sized(_, ref x1, ref x2), &BitType::Sized(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ListComprehension {
    pub span: ByteSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for ListComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body &&
        self.qualifiers == other.qualifiers
    }
}

#[derive(Debug, Clone)]
pub struct BinaryComprehension {
    pub span: ByteSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for BinaryComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body &&
        self.qualifiers == other.qualifiers
    }
}

// A generator of the form `LHS <- RHS`
#[derive(Debug, Clone)]
pub struct Generator {
    pub span: ByteSpan,
    pub pattern: Box<Expr>,
    pub expr: Box<Expr>,
}
impl PartialEq for Generator {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.expr == other.expr
    }
}

// A generator of the form `LHS <= RHS`
#[derive(Debug, Clone)]
pub struct BinaryGenerator {
    pub span: ByteSpan,
    pub pattern: Box<Expr>,
    pub expr: Box<Expr>,
}
impl PartialEq for BinaryGenerator {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.expr == other.expr
    }
}

// A sequence of expressions, e.g. begin expr1, .., exprN end
#[derive(Debug, Clone)]
pub struct Begin {
    pub span: ByteSpan,
    pub body: Vec<Expr>,
}
impl PartialEq for Begin {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

// Function application, e.g. foo(expr1, .., exprN)
#[derive(Debug, Clone)]
pub struct Apply {
    pub span: ByteSpan,
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}
impl PartialEq for Apply {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args
    }
}

// Remote, e.g. Foo:Bar
#[derive(Debug, Clone)]
pub struct Remote {
    pub span: ByteSpan,
    pub module: Box<Expr>,
    pub function: Box<Expr>,
}
impl PartialEq for Remote {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function
    }
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub span: ByteSpan,
    pub lhs: Box<Expr>,
    pub op: BinaryOp,
    pub rhs: Box<Expr>,
}
impl PartialEq for BinaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op &&
        self.lhs == other.lhs &&
        self.rhs == other.rhs
    }
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub span: ByteSpan,
    pub op: UnaryOp,
    pub operand: Box<Expr>,
}
impl PartialEq for UnaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.operand == other.operand
    }
}

#[derive(Debug, Clone)]
pub struct Match {
    pub span: ByteSpan,
    pub pattern: Box<Expr>,
    pub expr: Box<Expr>,
}
impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.expr == other.expr
    }
}

#[derive(Debug, Clone)]
pub struct If {
    pub span: ByteSpan,
    pub clauses: Vec<IfClause>,
}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
    }
}

/// Represents a single clause in an `if` expression
#[derive(Debug, Clone)]
pub struct IfClause {
    pub span: ByteSpan,
    pub conditions: Vec<Expr>,
    pub body: Vec<Expr>,
}
impl PartialEq for IfClause {
    fn eq(&self, other: &Self) -> bool {
        self.conditions == other.conditions && self.body == other.body
    }
}

#[derive(Debug, Clone)]
pub struct Catch {
    pub span: ByteSpan,
    pub expr: Box<Expr>,
}
impl PartialEq for Catch {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub span: ByteSpan,
    pub expr: Box<Expr>,
    pub clauses: Vec<Clause>,
}
impl PartialEq for Case {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr && self.clauses == other.clauses
    }
}

#[derive(Debug, Clone)]
pub struct Receive {
    pub span: ByteSpan,
    pub clauses: Option<Vec<Clause>>,
    pub after: Option<After>,
}
impl PartialEq for Receive {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses && self.after == other.after
    }
}

#[derive(Debug, Clone)]
pub struct Try {
    pub span: ByteSpan,
    pub exprs: Option<Vec<Expr>>,
    pub clauses: Option<Vec<Clause>>,
    pub catch_clauses: Option<Vec<TryClause>>,
    pub after: Option<Vec<Expr>>,
}
impl PartialEq for Try {
    fn eq(&self, other: &Self) -> bool {
        self.exprs == other.exprs &&
        self.clauses == other.clauses &&
        self.catch_clauses == other.catch_clauses &&
        self.after == other.after
    }
}

/// Represents a single `catch` clause in a `try` expression
#[derive(Debug, Clone)]
pub struct TryClause {
    pub span: ByteSpan,
    pub kind: Name,
    pub error: Expr,
    pub guard: Option<Vec<Guard>>,
    pub trace: Ident,
    pub body: Vec<Expr>,
}
impl PartialEq for TryClause {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind &&
        self.error == other.error &&
        self.guard == other.guard &&
        self.trace == other.trace &&
        self.body == other.body
    }
}

/// Represents the `after` clause of a `receive` expression
#[derive(Debug, Clone)]
pub struct After {
    pub span: ByteSpan,
    pub timeout: Box<Expr>,
    pub body: Vec<Expr>,
}
impl PartialEq for After {
    fn eq(&self, other: &Self) -> bool {
        self.timeout == other.timeout && self.body == other.body
    }
}

/// Represents a single match clause in a `case`, `try`, or `receive` expression
#[derive(Debug, Clone)]
pub struct Clause {
    pub span: ByteSpan,
    pub pattern: Expr,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}
impl PartialEq for Clause {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern &&
        self.guard == other.guard &&
        self.body == other.body
    }
}
