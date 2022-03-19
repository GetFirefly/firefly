//! Expressions
//!
//! See: [6.4 Expressions](http://erlang.org/doc/apps/erts/absform.html#id87350)
use super::*;

pub type LocalCall = common::LocalCall<Expression>;
pub type RemoteCall = common::RemoteCall<Expression>;
pub type Match = common::Match<Pattern, Expression>;
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
    Integer(Box<Integer>),
    Float(Box<Float>),
    String(Box<Str>),
    Char(Box<Char>),
    Atom(Box<Atom>),
    Match(Box<Match>),
    Var(Box<Var>),
    Tuple(Box<Tuple>),
    Nil(Box<Nil>),
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
    InternalFun(Box<InternalFun>),
    ExternalFun(Box<ExternalFun>),
    AnonymousFun(Box<AnonymousFun>),
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
impl_from!(Expression::InternalFun(InternalFun));
impl_from!(Expression::ExternalFun(ExternalFun));
impl_from!(Expression::AnonymousFun(AnonymousFun));
impl Node for Expression {
    fn line(&self) -> LineNum {
        match *self {
            Self::Integer(ref x) => x.line(),
            Self::Float(ref x) => x.line(),
            Self::String(ref x) => x.line(),
            Self::Char(ref x) => x.line(),
            Self::Atom(ref x) => x.line(),
            Self::Match(ref x) => x.line(),
            Self::Var(ref x) => x.line(),
            Self::Tuple(ref x) => x.line(),
            Self::Nil(ref x) => x.line(),
            Self::Cons(ref x) => x.line(),
            Self::Binary(ref x) => x.line(),
            Self::UnaryOp(ref x) => x.line(),
            Self::BinaryOp(ref x) => x.line(),
            Self::Record(ref x) => x.line(),
            Self::RecordIndex(ref x) => x.line(),
            Self::Map(ref x) => x.line(),
            Self::Catch(ref x) => x.line(),
            Self::LocalCall(ref x) => x.line(),
            Self::RemoteCall(ref x) => x.line(),
            Self::Comprehension(ref x) => x.line(),
            Self::Block(ref x) => x.line(),
            Self::If(ref x) => x.line(),
            Self::Case(ref x) => x.line(),
            Self::Try(ref x) => x.line(),
            Self::Receive(ref x) => x.line(),
            Self::InternalFun(ref x) => x.line(),
            Self::ExternalFun(ref x) => x.line(),
            Self::AnonymousFun(ref x) => x.line(),
        }
    }
}
impl Expression {
    pub fn atom(line: LineNum, name: String) -> Self {
        Self::Atom(Box::new(Atom::new(line, name)))
    }
}

#[derive(Debug, Clone)]
pub struct Catch {
    pub line: LineNum,
    pub expr: Expression,
}
impl Catch {
    pub fn new(line: LineNum, expr: Expression) -> Self {
        Catch { line, expr }
    }
}
impl Node for Catch {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct If {
    pub line: LineNum,
    pub clauses: Vec<Clause>,
}
impl If {
    pub fn new(line: LineNum, clauses: Vec<Clause>) -> Self {
        If { line, clauses }
    }
}
impl Node for If {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub line: LineNum,
    pub expr: Expression,
    pub clauses: Vec<Clause>,
}
impl Case {
    pub fn new(line: LineNum, expr: Expression, clauses: Vec<Clause>) -> Self {
        Case {
            line,
            expr,
            clauses,
        }
    }
}
impl Node for Case {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Try {
    pub line: LineNum,
    pub body: Vec<Expression>,
    pub case_clauses: Vec<Clause>,
    pub catch_clauses: Vec<Clause>,
    pub after: Vec<Expression>,
}
impl Try {
    pub fn new(
        line: LineNum,
        body: Vec<Expression>,
        case_clauses: Vec<Clause>,
        catch_clauses: Vec<Clause>,
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
impl Node for Try {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Receive {
    pub line: LineNum,
    pub clauses: Vec<Clause>,
    pub timeout: Option<Expression>,
    pub after: Vec<Expression>,
}
impl Receive {
    pub fn new(line: LineNum, clauses: Vec<Clause>) -> Self {
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
impl Node for Receive {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub line: LineNum,
    pub body: Vec<Expression>,
}
impl Block {
    pub fn new(line: LineNum, body: Vec<Expression>) -> Self {
        Block { line, body }
    }
}
impl Node for Block {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct Comprehension {
    pub line: LineNum,
    pub is_list: bool,
    pub expr: Expression,
    pub qualifiers: Vec<Qualifier>,
}
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
impl Node for Comprehension {
    fn line(&self) -> LineNum {
        self.line
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
    pub pattern: Pattern,
    pub expr: Expression,
}
impl Generator {
    pub fn new(line: LineNum, pattern: Pattern, expr: Expression) -> Self {
        Generator {
            line,
            pattern,
            expr,
        }
    }
}
impl Node for Generator {
    fn line(&self) -> LineNum {
        self.line
    }
}

#[derive(Debug, Clone)]
pub struct AnonymousFun {
    pub line: LineNum,
    pub name: Option<String>,
    pub clauses: Vec<Clause>,
}
impl AnonymousFun {
    pub fn new(line: LineNum, clauses: Vec<Clause>) -> Self {
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
impl Node for AnonymousFun {
    fn line(&self) -> LineNum {
        self.line
    }
}
