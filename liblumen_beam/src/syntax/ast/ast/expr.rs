//! Expressions
//!
//! See: [6.4 Expressions](http://erlang.org/doc/apps/erts/absform.html#id87350)
use super::clause;
use super::common;
use super::literal;
use super::pat;
use super::{LineNum, Node};

pub type LocalCall = common::LocalCall<Expression>;
pub type RemoteCall = common::RemoteCall<Expression>;
pub type Match = common::Match<pat::Pattern, Expression>;
pub type Tuple = common::Tuple<Expression>;
pub type Cons = common::Cons<Expression>;
pub type Binary = common::Binary<Expression>;
pub type UnaryOp = common::UnaryOp<Expression>;
pub type BinaryOp = common::BinaryOp<Expression>;
pub type Record = common::Record<Expression>;
pub type RecordIndex = common::RecordIndex<Expression>;
pub type Map = common::Map<Expression>;

#[derive(Debug, Clone)]
pub enum Expression {
    Integer(Box<literal::Integer>),
    Float(Box<literal::Float>),
    String(Box<literal::Str>),
    Char(Box<literal::Char>),
    Atom(Box<literal::Atom>),
    Match(Box<Match>),
    Var(Box<common::Var>),
    Tuple(Box<Tuple>),
    Nil(Box<common::Nil>),
    Cons(Box<Cons>),
    Binary(Binary),
    UnaryOp(Box<UnaryOp>),
    BinaryOp(Box<BinaryOp>),
    Record(Box<Record>),
    RecordIndex(Box<RecordIndex>),
    Map(Box<Map>),
    Catch(Box<Catch>),
    LocalCall(Box<LocalCall>),
    RemoteCall(Box<RemoteCall>),
    Comprehension(Box<Comprehension>),
    Block(Box<Block>),
    If(Box<If>),
    Case(Box<Case>),
    Try(Box<Try>),
    Receive(Box<Receive>),
    InternalFun(Box<common::InternalFun>),
    ExternalFun(Box<common::ExternalFun>),
    AnonymousFun(Box<AnonymousFun>),
}
impl_from!(Expression::Integer(literal::Integer));
impl_from!(Expression::Float(literal::Float));
impl_from!(Expression::String(literal::Str));
impl_from!(Expression::Char(literal::Char));
impl_from!(Expression::Atom(literal::Atom));
impl_from!(Expression::Match(Match));
impl_from!(Expression::Var(common::Var));
impl_from!(Expression::Tuple(Tuple));
impl_from!(Expression::Nil(common::Nil));
impl_from!(Expression::Cons(Cons));
impl_from!(Expression::Binary(Binary));
impl_from!(Expression::UnaryOp(UnaryOp));
impl_from!(Expression::BinaryOp(BinaryOp));
impl_from!(Expression::Record(Record));
impl_from!(Expression::RecordIndex(RecordIndex));
impl_from!(Expression::Map(Map));
impl_from!(Expression::Catch(Catch));
impl_from!(Expression::LocalCall(LocalCall));
impl_from!(Expression::RemoteCall(RemoteCall));
impl_from!(Expression::Comprehension(Comprehension));
impl_from!(Expression::Block(Block));
impl_from!(Expression::If(If));
impl_from!(Expression::Case(Case));
impl_from!(Expression::Try(Try));
impl_from!(Expression::Receive(Receive));
impl_from!(Expression::InternalFun(common::InternalFun));
impl_from!(Expression::ExternalFun(common::ExternalFun));
impl_from!(Expression::AnonymousFun(AnonymousFun));
impl Node for Expression {
    fn line(&self) -> LineNum {
        match *self {
            Expression::Integer(ref x) => x.line(),
            Expression::Float(ref x) => x.line(),
            Expression::String(ref x) => x.line(),
            Expression::Char(ref x) => x.line(),
            Expression::Atom(ref x) => x.line(),
            Expression::Match(ref x) => x.line(),
            Expression::Var(ref x) => x.line(),
            Expression::Tuple(ref x) => x.line(),
            Expression::Nil(ref x) => x.line(),
            Expression::Cons(ref x) => x.line(),
            Expression::Binary(ref x) => x.line(),
            Expression::UnaryOp(ref x) => x.line(),
            Expression::BinaryOp(ref x) => x.line(),
            Expression::Record(ref x) => x.line(),
            Expression::RecordIndex(ref x) => x.line(),
            Expression::Map(ref x) => x.line(),
            Expression::Catch(ref x) => x.line(),
            Expression::LocalCall(ref x) => x.line(),
            Expression::RemoteCall(ref x) => x.line(),
            Expression::Comprehension(ref x) => x.line(),
            Expression::Block(ref x) => x.line(),
            Expression::If(ref x) => x.line(),
            Expression::Case(ref x) => x.line(),
            Expression::Try(ref x) => x.line(),
            Expression::Receive(ref x) => x.line(),
            Expression::InternalFun(ref x) => x.line(),
            Expression::ExternalFun(ref x) => x.line(),
            Expression::AnonymousFun(ref x) => x.line(),
        }
    }
}
impl Expression {
    pub fn atom(line: LineNum, name: String) -> Self {
        Expression::Atom(Box::new(literal::Atom::new(line, name)))
    }
}

#[derive(Debug, Clone)]
pub struct Catch {
    pub line: LineNum,
    pub expr: Expression,
}
impl_node!(Catch);
impl Catch {
    pub fn new(line: LineNum, expr: Expression) -> Self {
        Catch { line, expr }
    }
}

#[derive(Debug, Clone)]
pub struct If {
    pub line: LineNum,
    pub clauses: Vec<clause::Clause>,
}
impl_node!(If);
impl If {
    pub fn new(line: LineNum, clauses: Vec<clause::Clause>) -> Self {
        If { line, clauses }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub line: LineNum,
    pub expr: Expression,
    pub clauses: Vec<clause::Clause>,
}
impl_node!(Case);
impl Case {
    pub fn new(line: LineNum, expr: Expression, clauses: Vec<clause::Clause>) -> Self {
        Case {
            line,
            expr,
            clauses,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Try {
    pub line: LineNum,
    pub body: Vec<Expression>,
    pub case_clauses: Vec<clause::Clause>,
    pub catch_clauses: Vec<clause::Clause>,
    pub after: Vec<Expression>,
}
impl_node!(Try);
impl Try {
    pub fn new(
        line: LineNum,
        body: Vec<Expression>,
        case_clauses: Vec<clause::Clause>,
        catch_clauses: Vec<clause::Clause>,
        after: Vec<Expression>,
    ) -> Self {
        Try {
            line,
            body,
            case_clauses,
            catch_clauses,
            after,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Receive {
    pub line: LineNum,
    pub clauses: Vec<clause::Clause>,
    pub timeout: Option<Expression>,
    pub after: Vec<Expression>,
}
impl_node!(Receive);
impl Receive {
    pub fn new(line: LineNum, clauses: Vec<clause::Clause>) -> Self {
        Receive {
            line,
            clauses,
            timeout: None,
            after: Vec::new(),
        }
    }
    pub fn timeout(mut self, timeout: Expression) -> Self {
        self.timeout = Some(timeout);
        self
    }
    pub fn after(mut self, after: Vec<Expression>) -> Self {
        self.after = after;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub line: LineNum,
    pub body: Vec<Expression>,
}
impl_node!(Block);
impl Block {
    pub fn new(line: LineNum, body: Vec<Expression>) -> Self {
        Block { line, body }
    }
}

#[derive(Debug, Clone)]
pub struct Comprehension {
    pub line: LineNum,
    pub is_list: bool,
    pub expr: Expression,
    pub qualifiers: Vec<Qualifier>,
}
impl_node!(Comprehension);
impl Comprehension {
    pub fn new(line: LineNum, is_list: bool, expr: Expression, qualifiers: Vec<Qualifier>) -> Self {
        Comprehension {
            line,
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
    pub line: LineNum,
    pub pattern: pat::Pattern,
    pub expr: Expression,
}
impl_node!(Generator);
impl Generator {
    pub fn new(line: LineNum, pattern: pat::Pattern, expr: Expression) -> Self {
        Generator {
            line,
            pattern,
            expr,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnonymousFun {
    pub line: LineNum,
    pub name: Option<String>,
    pub clauses: Vec<clause::Clause>,
}
impl_node!(AnonymousFun);
impl AnonymousFun {
    pub fn new(line: LineNum, clauses: Vec<clause::Clause>) -> Self {
        AnonymousFun {
            line,
            name: None,
            clauses,
        }
    }
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}
