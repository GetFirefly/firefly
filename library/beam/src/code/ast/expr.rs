//! Expressions
//!
//! See: [6.4 Expressions](http://erlang.org/doc/apps/erts/absform.html#id87350)
use firefly_intern::Symbol;

use super::*;

#[derive(Debug, Clone)]
pub enum Expression {
    Integer(Box<Integer>),
    Float(Float),
    String(Box<Str>),
    Char(Char),
    Atom(Atom),
    Match(Box<Match>),
    Var(Box<Var>),
    Tuple(Box<Tuple>),
    Nil(Box<Nil>),
    Cons(Box<Cons>),
    Binary(Box<Binary>),
    UnaryOp(Box<UnaryOp>),
    BinaryOp(Box<BinaryOp>),
    Record(Box<Record>),
    RecordIndex(Box<RecordIndex>),
    RecordAccess(Box<RecordAccess>),
    Map(Box<Map>),
    Catch(Box<Catch>),
    Call(Box<Call>),
    Comprehension(Box<Comprehension>),
    Block(Box<Block>),
    If(Box<If>),
    Case(Box<Case>),
    Try(Box<Try>),
    Receive(Box<Receive>),
    InternalFun(Box<InternalFun>),
    ExternalFun(Box<ExternalFun>),
    AnonymousFun(Box<AnonymousFun>),
    Remote(Box<Remote>),
}
impl_from!(Expression::Integer(Integer));
impl_from!(Expression::Float(Float));
impl_from!(Expression::String(Str));
impl_from!(Expression::Char(Char));
impl_from!(Expression::Atom(Atom));
impl_from!(Expression::Match(Match));
impl_from!(Expression::Var(Var));
impl_from!(Expression::Tuple(Tuple));
impl_from!(Expression::Nil(Nil));
impl_from!(Expression::Cons(Cons));
impl_from!(Expression::Binary(Binary));
impl_from!(Expression::UnaryOp(UnaryOp));
impl_from!(Expression::BinaryOp(BinaryOp));
impl_from!(Expression::Record(Record));
impl_from!(Expression::RecordIndex(RecordIndex));
impl_from!(Expression::RecordAccess(RecordAccess));
impl_from!(Expression::Map(Map));
impl_from!(Expression::Catch(Catch));
impl_from!(Expression::Call(Call));
impl_from!(Expression::Comprehension(Comprehension));
impl_from!(Expression::Block(Block));
impl_from!(Expression::If(If));
impl_from!(Expression::Case(Case));
impl_from!(Expression::Try(Try));
impl_from!(Expression::Receive(Receive));
impl_from!(Expression::InternalFun(InternalFun));
impl_from!(Expression::ExternalFun(ExternalFun));
impl_from!(Expression::AnonymousFun(AnonymousFun));
impl_from!(Expression::Remote(Remote));
impl Node for Expression {
    fn loc(&self) -> Location {
        match self {
            Self::Integer(ref x) => x.loc(),
            Self::Float(ref x) => x.loc(),
            Self::String(ref x) => x.loc(),
            Self::Char(ref x) => x.loc(),
            Self::Atom(ref x) => x.loc(),
            Self::Match(ref x) => x.loc(),
            Self::Var(ref x) => x.loc(),
            Self::Tuple(ref x) => x.loc(),
            Self::Nil(ref x) => x.loc(),
            Self::Cons(ref x) => x.loc(),
            Self::Binary(ref x) => x.loc(),
            Self::UnaryOp(ref x) => x.loc(),
            Self::BinaryOp(ref x) => x.loc(),
            Self::Record(ref x) => x.loc(),
            Self::RecordIndex(ref x) => x.loc(),
            Self::RecordAccess(ref x) => x.loc(),
            Self::Map(ref x) => x.loc(),
            Self::Catch(ref x) => x.loc(),
            Self::Call(ref x) => x.loc(),
            Self::Comprehension(ref x) => x.loc(),
            Self::Block(ref x) => x.loc(),
            Self::If(ref x) => x.loc(),
            Self::Case(ref x) => x.loc(),
            Self::Try(ref x) => x.loc(),
            Self::Receive(ref x) => x.loc(),
            Self::InternalFun(ref x) => x.loc(),
            Self::ExternalFun(ref x) => x.loc(),
            Self::AnonymousFun(ref x) => x.loc(),
            Self::Remote(ref x) => x.loc(),
        }
    }
}
impl Expression {
    pub fn atom(loc: Location, name: Symbol) -> Self {
        Self::Atom(Atom::new(loc, name))
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Nil {
    pub loc: Location,
}
impl_node!(Nil);
impl Nil {
    pub fn new(loc: Location) -> Self {
        Self { loc }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Var {
    pub loc: Location,
    pub name: Symbol,
}
impl_node!(Var);
impl Var {
    pub fn new(loc: Location, name: Symbol) -> Self {
        Var { loc, name }
    }

    pub fn is_wildcard(&self) -> bool {
        self.name == "_"
    }
}

#[derive(Debug, Clone)]
pub struct Match {
    pub loc: Location,
    pub left: Expression,
    pub right: Expression,
}
impl_node!(Match);
impl Match {
    pub fn new(loc: Location, left: Expression, right: Expression) -> Self {
        Self { loc, left, right }
    }
}

#[derive(Debug, Clone)]
pub struct Tuple {
    pub loc: Location,
    pub elements: Vec<Expression>,
}
impl_node!(Tuple);
impl Tuple {
    pub fn new(loc: Location, elements: Vec<Expression>) -> Self {
        Self { loc, elements }
    }
}

#[derive(Debug, Clone)]
pub struct Cons {
    pub loc: Location,
    pub head: Expression,
    pub tail: Expression,
}
impl_node!(Cons);
impl Cons {
    pub fn new(loc: Location, head: Expression, tail: Expression) -> Self {
        Self { loc, head, tail }
    }
}

#[derive(Debug, Clone)]
pub struct Binary {
    pub loc: Location,
    pub elements: Vec<BinElement>,
}
impl_node!(Binary);
impl Binary {
    pub fn new(loc: Location, elements: Vec<BinElement>) -> Self {
        Self { loc, elements }
    }
}

#[derive(Debug, Clone)]
pub struct BinElement {
    pub loc: Location,
    pub element: Expression,
    pub size: Option<Expression>,
    pub tsl: Option<Vec<BinElementTypeSpec>>,
}
impl_node!(BinElement);
impl BinElement {
    pub fn new(
        loc: Location,
        element: Expression,
        size: Option<Expression>,
        tsl: Option<Vec<BinElementTypeSpec>>,
    ) -> Self {
        Self {
            loc,
            element,
            size,
            tsl,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinElementTypeSpec {
    pub name: Symbol,
    pub value: Option<u64>,
}
impl BinElementTypeSpec {
    pub fn new(name: Symbol, value: Option<u64>) -> Self {
        Self { name, value }
    }
}

#[derive(Debug, Clone)]
pub struct UnaryOp {
    pub loc: Location,
    pub operator: Symbol,
    pub operand: Expression,
}
impl_node!(UnaryOp);
impl UnaryOp {
    pub fn new(loc: Location, operator: Symbol, operand: Expression) -> Self {
        Self {
            loc,
            operator,
            operand,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub loc: Location,
    pub operator: Symbol,
    pub left_operand: Expression,
    pub right_operand: Expression,
}
impl_node!(BinaryOp);
impl BinaryOp {
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
pub struct Record {
    pub loc: Location,
    pub base: Option<Expression>,
    pub name: Symbol,
    pub fields: Vec<RecordField>,
}
impl_node!(Record);
impl Record {
    pub fn new(
        loc: Location,
        base: Option<Expression>,
        name: Symbol,
        fields: Vec<RecordField>,
    ) -> Self {
        Self {
            loc,
            base,
            name,
            fields,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordField {
    pub loc: Location,
    pub name: Option<Symbol>, // `None` means `_` (i.e., default value)
    pub value: Expression,
}
impl_node!(RecordField);
impl RecordField {
    pub fn new(loc: Location, name: Option<Symbol>, value: Expression) -> Self {
        Self { loc, name, value }
    }
}

#[derive(Debug, Clone)]
pub struct RecordAccess {
    pub loc: Location,
    pub base: Expression,
    pub record: Symbol,
    pub field: Symbol,
}
impl_node!(RecordAccess);
impl RecordAccess {
    pub fn new(loc: Location, base: Expression, record: Symbol, field: Symbol) -> Self {
        Self {
            loc,
            base,
            record,
            field,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordIndex {
    pub loc: Location,
    pub record: Symbol,
    pub field: Symbol,
}
impl_node!(RecordIndex);
impl RecordIndex {
    pub fn new(loc: Location, record: Symbol, field: Symbol) -> Self {
        Self { loc, record, field }
    }
}

#[derive(Debug, Clone)]
pub struct Map {
    pub loc: Location,
    pub base: Option<Expression>,
    pub pairs: Vec<MapPair>,
}
impl_node!(Map);
impl Map {
    pub fn new(loc: Location, base: Option<Expression>, pairs: Vec<MapPair>) -> Self {
        Self { loc, base, pairs }
    }
}

#[derive(Debug, Clone)]
pub struct MapPair {
    pub loc: Location,
    pub is_assoc: bool,
    pub key: Expression,
    pub value: Expression,
}
impl_node!(MapPair);
impl MapPair {
    pub fn new(loc: Location, is_assoc: bool, key: Expression, value: Expression) -> Self {
        Self {
            loc,
            is_assoc,
            key,
            value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InternalFun {
    pub loc: Location,
    pub function: Symbol,
    pub arity: Arity,
}
impl_node!(InternalFun);
impl InternalFun {
    pub fn new(loc: Location, function: Symbol, arity: Arity) -> Self {
        Self {
            loc,
            function,
            arity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExternalFun {
    pub loc: Location,
    pub module: Expression,
    pub function: Expression,
    pub arity: Expression,
}
impl_node!(ExternalFun);
impl ExternalFun {
    pub fn new(loc: Location, module: Expression, function: Expression, arity: Expression) -> Self {
        Self {
            loc,
            module,
            function,
            arity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Catch {
    pub loc: Location,
    pub expr: Expression,
}
impl_node!(Catch);
impl Catch {
    pub fn new(loc: Location, expr: Expression) -> Self {
        Self { loc, expr }
    }
}

#[derive(Debug, Clone)]
pub struct If {
    pub loc: Location,
    pub clauses: Vec<Clause>,
}
impl_node!(If);
impl If {
    pub fn new(loc: Location, clauses: Vec<Clause>) -> Self {
        Self { loc, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub loc: Location,
    pub expr: Expression,
    pub clauses: Vec<Clause>,
}
impl_node!(Case);
impl Case {
    pub fn new(loc: Location, expr: Expression, clauses: Vec<Clause>) -> Self {
        Self { loc, expr, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct Try {
    pub loc: Location,
    pub body: Vec<Expression>,
    pub case_clauses: Vec<Clause>,
    pub catch_clauses: Vec<Clause>,
    pub after: Vec<Expression>,
}
impl_node!(Try);
impl Try {
    pub fn new(
        loc: Location,
        body: Vec<Expression>,
        case_clauses: Vec<Clause>,
        catch_clauses: Vec<Clause>,
        after: Vec<Expression>,
    ) -> Self {
        Self {
            loc,
            body,
            case_clauses,
            catch_clauses,
            after,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Receive {
    pub loc: Location,
    pub clauses: Vec<Clause>,
    pub timeout: Option<Expression>,
    pub after: Vec<Expression>,
}
impl_node!(Receive);
impl Receive {
    pub fn new(
        loc: Location,
        clauses: Vec<Clause>,
        timeout: Option<Expression>,
        after: Vec<Expression>,
    ) -> Self {
        Self {
            loc,
            clauses,
            timeout,
            after,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub loc: Location,
    pub body: Vec<Expression>,
}
impl_node!(Block);
impl Block {
    pub fn new(loc: Location, body: Vec<Expression>) -> Self {
        Self { loc, body }
    }
}

#[derive(Debug, Clone)]
pub struct Comprehension {
    pub loc: Location,
    pub is_list: bool,
    pub expr: Expression,
    pub qualifiers: Vec<Qualifier>,
}
impl_node!(Comprehension);
impl Comprehension {
    pub fn new(loc: Location, is_list: bool, expr: Expression, qualifiers: Vec<Qualifier>) -> Self {
        Self {
            loc,
            is_list,
            expr,
            qualifiers,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Qualifier {
    Generator(Generator),
    BitStringGenerator(Generator),
    Filter(Expression),
}

#[derive(Debug, Clone)]
pub struct Generator {
    pub loc: Location,
    pub pattern: Expression,
    pub expr: Expression,
}
impl_node!(Generator);
impl Generator {
    pub fn new(loc: Location, pattern: Expression, expr: Expression) -> Self {
        Self { loc, pattern, expr }
    }
}

#[derive(Debug, Clone)]
pub struct AnonymousFun {
    pub loc: Location,
    pub name: Option<Symbol>,
    pub clauses: Vec<Clause>,
}
impl_node!(AnonymousFun);
impl AnonymousFun {
    pub fn new(loc: Location, name: Option<Symbol>, clauses: Vec<Clause>) -> Self {
        Self { loc, name, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub loc: Location,
    pub callee: Expression,
    pub args: Vec<Expression>,
}
impl_node!(Call);
impl Call {
    pub fn new(loc: Location, callee: Expression, args: Vec<Expression>) -> Self {
        Self { loc, callee, args }
    }
}

#[derive(Debug, Clone)]
pub struct Remote {
    pub loc: Location,
    pub module: Expression,
    pub function: Expression,
}
impl_node!(Remote);
impl Remote {
    pub fn new(loc: Location, module: Expression, function: Expression) -> Self {
        Self {
            loc,
            module,
            function,
        }
    }
}
