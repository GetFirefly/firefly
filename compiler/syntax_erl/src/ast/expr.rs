use std::cmp::Ordering;

use liblumen_binary::BinaryEntrySpecifier;
use liblumen_diagnostics::SourceSpan;
use liblumen_intern::{symbols, Symbol};
use liblumen_number::{Float, Integer, Number};
use liblumen_syntax_core::{self as syntax_core};

use super::{BinaryOp, Ident, UnaryOp};
use super::{Fun, FunctionName, Guard, Name, Type};

use crate::evaluator::{self, EvalError};
use crate::lexer::DelayedSubstitution;

/// The set of all possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // An identifier/variable/function reference
    Var(Var),
    Literal(Literal),
    FunctionName(FunctionName),
    // Delayed substitution of macro
    DelayedSubstitution(SourceSpan, DelayedSubstitution),
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
    Fun(Fun),
}
impl Expr {
    pub fn span(&self) -> SourceSpan {
        match self {
            &Expr::Var(Var(Ident { ref span, .. })) => span.clone(),
            &Expr::Literal(ref lit) => lit.span(),
            &Expr::FunctionName(ref name) => name.span(),
            &Expr::DelayedSubstitution(ref span, _) => *span,
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

    /// Returns true if this expression is one that is sensitive to imperative assignment
    pub fn is_block_like(&self) -> bool {
        match self {
            Expr::Match(Match { ref expr, .. }) => expr.is_block_like(),
            Expr::Begin(_) | Expr::If(_) | Expr::Case(_) => true,
            _ => false,
        }
    }

    /// If this expression is an atom, this function returns the Ident
    /// backing the atom value. This is a common request in the compiler,
    /// hence its presence here
    pub fn as_atom(&self) -> Option<Ident> {
        match self {
            Expr::Literal(Literal::Atom(a)) => Some(*a),
            _ => None,
        }
    }

    /// Returns `Some(bool)` if the expression represents a literal boolean, otherwise None
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Expr::Literal(lit) => lit.as_boolean(),
            _ => None,
        }
    }

    pub fn as_var(&self) -> Option<Var> {
        match self {
            Expr::Var(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            Expr::Literal(ref lit) => Some(lit),
            _ => None,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Var(pub Ident);
impl Var {
    #[inline]
    pub fn sym(&self) -> Symbol {
        self.0.name
    }

    #[inline]
    pub fn span(&self) -> SourceSpan {
        self.0.span
    }

    #[inline]
    pub fn is_wildcard(&self) -> bool {
        self.0.name == symbols::WildcardMatch
    }
}
impl From<Ident> for Var {
    fn from(i: Ident) -> Self {
        Self(i)
    }
}

#[derive(Debug, Clone)]
pub struct Nil(pub SourceSpan);
impl Nil {
    #[inline(always)]
    pub fn span(&self) -> SourceSpan {
        self.0
    }
}
impl PartialEq for Nil {
    fn eq(&self, _: &Self) -> bool {
        return true;
    }
}
impl Eq for Nil {}

#[derive(Debug, Clone)]
pub struct Cons {
    pub span: SourceSpan,
    pub head: Box<Expr>,
    pub tail: Box<Expr>,
}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub span: SourceSpan,
    pub elements: Vec<Expr>,
}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

#[derive(Debug, Clone)]
pub struct Map {
    pub span: SourceSpan,
    pub fields: Vec<MapField>,
}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

// Updating fields on an existing map, e.g. `Map#{field1 = value1}.`
#[derive(Debug, Clone)]
pub struct MapUpdate {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub map: Box<Expr>,
    pub fields: Vec<MapField>,
}
impl PartialEq for MapProjection {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map && self.fields == other.fields
    }
}

/// Maps can have two different types of field assignment:
///
/// * assoc - inserts or updates the given key with the given value
/// * exact - updates the given key with the given value, or produces an error
#[derive(Debug, Clone)]
pub enum MapField {
    Assoc {
        span: SourceSpan,
        key: Expr,
        value: Expr,
    },
    Exact {
        span: SourceSpan,
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

    pub fn span(&self) -> SourceSpan {
        match self {
            MapField::Assoc { span, .. } => *span,
            MapField::Exact { span, .. } => *span,
        }
    }
}
impl PartialEq for MapField {
    fn eq(&self, other: &Self) -> bool {
        (self.key() == other.key()) && (self.value() == other.value())
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
    Char(SourceSpan, char),
    Integer(SourceSpan, Integer),
    Float(SourceSpan, Float),
}
impl Literal {
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Atom(id) => match id.name {
                symbols::True => Some(true),
                symbols::False => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn span(&self) -> SourceSpan {
        match self {
            &Literal::Atom(Ident { ref span, .. }) => span.clone(),
            &Literal::String(Ident { ref span, .. }) => span.clone(),
            &Literal::Char(span, _) => span.clone(),
            &Literal::Integer(span, _) => span.clone(),
            &Literal::Float(span, _) => span.clone(),
        }
    }
}
impl PartialEq for Literal {
    fn eq(&self, other: &Literal) -> bool {
        match (self, other) {
            (&Literal::Atom(ref lhs), &Literal::Atom(ref rhs)) => lhs == rhs,
            (&Literal::Atom(_), _) => false,
            (_, &Literal::Atom(_)) => false,
            (&Literal::String(ref lhs), &Literal::String(ref rhs)) => lhs == rhs,
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
            (Literal::String(ref lhs), Literal::String(ref rhs)) => lhs.partial_cmp(rhs),
            (Literal::String(_), _) => Some(Ordering::Greater),
            (_, Literal::String(_)) => Some(Ordering::Less),
            (Literal::Atom(ref lhs), Literal::Atom(ref rhs)) => lhs.partial_cmp(rhs),
            (Literal::Atom(_), _) => Some(Ordering::Greater),
            (_, Literal::Atom(_)) => Some(Ordering::Less),

            (
                l @ (Literal::Integer(_, _) | Literal::Float(_, _) | Literal::Char(_, _)),
                r @ (Literal::Integer(_, _) | Literal::Float(_, _) | Literal::Char(_, _)),
            ) => {
                let to_num = |lit: &Literal| match lit {
                    Literal::Integer(_, x) => x.clone().into(),
                    Literal::Float(_, x) => x.clone().into(),
                    Literal::Char(_, x) => {
                        let int: Integer = (*x).into();
                        int.into()
                    }
                    _ => unreachable!(),
                };

                let ln: Number = to_num(l);
                let rn: Number = to_num(r);

                ln.partial_cmp(&rn)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordAccess {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.field == other.field && self.record == other.record
    }
}

// Referencing a record fields index, e.g. #myrec.field1
#[derive(Debug, Clone)]
pub struct RecordIndex {
    pub span: SourceSpan,
    pub name: Ident,
    pub field: Ident,
}
impl PartialEq for RecordIndex {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.field == other.field
    }
}

// Update a record field value, e.g. Expr#myrec{field1=ValueExpr}
#[derive(Debug, Clone)]
pub struct RecordUpdate {
    pub span: SourceSpan,
    pub record: Box<Expr>,
    pub name: Ident,
    pub updates: Vec<RecordField>,
}
impl PartialEq for RecordUpdate {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.record == other.record && self.updates == other.updates
    }
}

/// Record fields always have a name, but both default value and type
/// are optional in a record definition. When instantiating a record,
/// if no value is given for a field, and no default is given,
/// then `undefined` is the default.
#[derive(Debug, Clone)]
pub struct RecordField {
    pub span: SourceSpan,
    pub name: Ident,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub bit_expr: Expr,
    pub bit_size: Option<Expr>,
    pub specifier: Option<BinaryEntrySpecifier>,
}
impl PartialEq for BinaryElement {
    fn eq(&self, other: &Self) -> bool {
        (self.bit_expr == other.bit_expr)
            && (self.bit_size == other.bit_size)
            && (self.specifier == other.specifier)
    }
}

/// A bit type can come in the form `Type` or `Type:Size`
#[derive(Debug, Clone)]
pub enum BitType {
    Name(SourceSpan, Ident),
    Sized(SourceSpan, Ident, i64),
}
impl BitType {
    pub fn span(&self) -> SourceSpan {
        match self {
            BitType::Name(span, _) => *span,
            BitType::Sized(span, _, _) => *span,
        }
    }
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
    pub span: SourceSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for ListComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.qualifiers == other.qualifiers
    }
}

#[derive(Debug, Clone)]
pub struct BinaryComprehension {
    pub span: SourceSpan,
    pub body: Box<Expr>,
    pub qualifiers: Vec<Expr>,
}
impl PartialEq for BinaryComprehension {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.qualifiers == other.qualifiers
    }
}

// A generator of the form `LHS <- RHS`
#[derive(Debug, Clone)]
pub struct Generator {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub module: Box<Expr>,
    pub function: Box<Expr>,
}
impl Remote {
    /// Try to resolve this remote expression to a constant function reference of the given arity
    pub fn try_eval(&self, arity: u8) -> Result<syntax_core::FunctionName, EvalError> {
        use crate::evaluator::Term;

        let span = self.span;
        let module = evaluator::eval_expr(self.module.as_ref(), None)?;
        let function = evaluator::eval_expr(self.function.as_ref(), None)?;
        match (module, function) {
            (Term::Atom(m), Term::Atom(f)) => Ok(syntax_core::FunctionName::new(m, f, arity)),
            _ => Err(EvalError::InvalidConstExpression { span }),
        }
    }
}
impl PartialEq for Remote {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function
    }
}

#[derive(Debug, Clone)]
pub struct BinaryExpr {
    pub span: SourceSpan,
    pub lhs: Box<Expr>,
    pub op: BinaryOp,
    pub rhs: Box<Expr>,
}
impl PartialEq for BinaryExpr {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.lhs == other.lhs && self.rhs == other.rhs
    }
}

#[derive(Debug, Clone)]
pub struct UnaryExpr {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub clauses: Vec<IfClause>,
}
impl If {
    /// Returns true if the last clause of the `if` is the literal boolean `true`
    pub fn has_wildcard_clause(&self) -> bool {
        self.clauses
            .last()
            .map(|clause| clause.is_wildcard())
            .unwrap_or(false)
    }
}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
    }
}

/// Represents a single clause in an `if` expression
#[derive(Debug, Clone)]
pub struct IfClause {
    pub span: SourceSpan,
    pub guards: Vec<Guard>,
    pub body: Vec<Expr>,
}
impl IfClause {
    pub fn is_wildcard(&self) -> bool {
        assert!(
            !self.guards.is_empty(),
            "invalid if clause, must have at least one guard expression"
        );
        if self.guards.len() > 1 {
            return false;
        }
        self.guards[0].as_boolean().unwrap_or(false)
    }
}
impl PartialEq for IfClause {
    fn eq(&self, other: &Self) -> bool {
        self.guards == other.guards && self.body == other.body
    }
}

#[derive(Debug, Clone)]
pub struct Catch {
    pub span: SourceSpan,
    pub expr: Box<Expr>,
}
impl PartialEq for Catch {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub exprs: Vec<Expr>,
    pub clauses: Option<Vec<Clause>>,
    pub catch_clauses: Option<Vec<TryClause>>,
    pub after: Option<Vec<Expr>>,
}
impl PartialEq for Try {
    fn eq(&self, other: &Self) -> bool {
        self.exprs == other.exprs
            && self.clauses == other.clauses
            && self.catch_clauses == other.catch_clauses
            && self.after == other.after
    }
}

/// Represents a single `catch` clause in a `try` expression
#[derive(Debug, Clone)]
pub struct TryClause {
    pub span: SourceSpan,
    pub kind: Name,
    pub error: Expr,
    pub guard: Option<Vec<Guard>>,
    pub trace: Ident,
    pub body: Vec<Expr>,
}
impl PartialEq for TryClause {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.error == other.error
            && self.guard == other.guard
            && self.trace == other.trace
            && self.body == other.body
    }
}

/// Represents the `after` clause of a `receive` expression
#[derive(Debug, Clone)]
pub struct After {
    pub span: SourceSpan,
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
    pub span: SourceSpan,
    pub pattern: Expr,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}
impl PartialEq for Clause {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.guard == other.guard && self.body == other.body
    }
}
