#![allow(missing_docs)]
use std::fmt;

use num::BigInt;

use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};

use crate::lexer::Ident;
use super::{ParseError, ParserError};

macro_rules! to_lalrpop_err (
    ($error:expr) => (lalrpop_util::ParseError::User { error: $error })
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionName {
    pub span: ByteSpan,
    pub module: Option<Ident>,
    pub function: Ident,
    pub arity: Option<usize>
}
impl FunctionName {
    pub fn span(&self) -> ByteSpan {
        self.span
    }

    pub fn from_clause(clause: &FunctionClause) -> FunctionName {
        let span = clause.span.clone();
        let function = clause.name.clone()
            .expect("cannot create FunctionName from anonymous FunctionClause");
        let arity = Some(clause.params.len());
        FunctionName {
            span,
            module: None,
            function,
            arity
        }
    }
}
impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionName { module: Some(ref m), ref function, arity: Some(a), .. } =>
                write!(f, "{}:{}/{}", m, function, a),
            FunctionName { module: Some(ref m), ref function, .. } =>
                write!(f, "{}:{}", m, function),
            FunctionName { ref function, arity: Some(a), .. } =>
                write!(f, "{}/{}", function, a),
            FunctionName { ref function, .. } =>
                write!(f, "{}", function),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Attribute(Attribute),
    Record(Record),
    Function(Function)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub span: ByteSpan,
    pub name: Ident,
    pub attributes: Vec<Attribute>,
    pub records: Vec<Record>,
    pub functions: Vec<Function>
}
impl Module {
    pub fn new(span: ByteSpan, name: Ident, body: Vec<TopLevel>) -> Self {
        let attributes = body.iter().filter_map(|b| {
            match b {
                TopLevel::Attribute(a) => Some(a.clone()),
                _ => None
            }
        }).collect();
        let records = body.iter().filter_map(|b| {
            match b {
                TopLevel::Record(r) => Some(r.clone()),
                _ => None
            }
        }).collect();
        let functions = body.iter().filter_map(|b| {
            match b {
                TopLevel::Function(f) => {
                    Some(f.clone())
                },
                _ => None
            }
        }).collect();
        Module {
            span,
            name,
            attributes,
            records,
            functions,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    // Types
    Type(Type),
    Opaque(Type),
    ImportType { span: ByteSpan, module: Ident, types: Vec<FunctionName> },
    Spec { span: ByteSpan, name: Ident, sig: Vec<TypeSig> },
    // Builtins
    Export { span: ByteSpan, exports: Vec<FunctionName> },
    Import { span: ByteSpan, imports: Vec<FunctionName> },
    Compile { span: ByteSpan, opts: Vec<ConstantExpr> },
    Vsn { span: ByteSpan, vsn: ConstantExpr },
    OnLoad { span: ByteSpan, fun: FunctionName },
    Callback { span: ByteSpan, module: Option<Ident>, fun: Ident, sig: Vec<TypeSig> },
    // Preprocessor Directives/Macros
    Macro { span: ByteSpan, name: Ident, args: Option<Vec<Ident>>, value: Expr },
    // User-defined attributes
    Custom { span: ByteSpan, name: Ident, value: ConstantExpr }
}
impl Attribute {
    pub fn new(span: ByteSpan, ident: Ident, values: Vec<ConstantExpr>) -> Result<Attribute, ParseError> {
        let arity = values.len();
        match ident.name.as_str().get() {
            "vsn" if arity == 1 => {
                Ok(Attribute::Vsn { span, vsn: values[0].clone() })
            }
            "vsn" => {
                Err(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_warning("invalid 'vsn' attribute")
                        .with_label(Label::new_primary(span)
                            .with_message(format!("expected one argument, but got {}", arity)))
                )))
            }
            "on_load" if arity == 1 => {
                let value = values[0].clone();
                if let ConstantExpr::FunctionName(fun) = value {
                    return Ok(Attribute::OnLoad { span, fun });
                }
                Err(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("invalid 'on_load' attribute")
                        .with_label(Label::new_primary(span)
                            .with_message(format!("expected function name, got {:?}", value)))
                )))
            }
            "on_load" => {
                Err(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("invalid 'on_load' attribute")
                        .with_label(Label::new_primary(span)
                            .with_message(format!("expected one argument, got {}", arity)))
                )))
            }
            _ => {
                unimplemented!()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Annotated { span: ByteSpan, name: Ident, ty: Box<Type> },
    Union { span: ByteSpan, lhs: Box<Type>, rhs: Box<Type> },
    Range(ByteSpan, Box<Type>, Box<Type>),
    BinaryOp { span: ByteSpan, lhs: Box<Type>, op: BinaryOp, rhs: Box<Type> },
    UnaryOp { span: ByteSpan, op: BinaryOp, rhs: Box<Type> },
    Var(Ident),
    Name(Ident),
    Generic { span: ByteSpan, name: Ident, params: Vec<Ident>, ty: Box<Type> },
    Remote { span: ByteSpan, module: Ident, fun: Ident, args: Vec<Ident> },
    Nil(ByteSpan),
    List(ByteSpan, Box<Type>),
    NonEmptyList(ByteSpan, Box<Type>),
    Map(ByteSpan, Vec<MapPairType>),
    Tuple(ByteSpan, Vec<Type>),
    Record { span: ByteSpan, name: Ident, fields: Vec<(ByteSpan, Ident, Type)> },
    Binary { span: ByteSpan, head: Box<Type>, tail: Box<Type> },
    Integer(ByteSpan, i64),
    Char(ByteSpan, char),
    Fun(FunType),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeSig {
    sig: FunType,
    constraints: Vec<TypeGuard>
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunType {
    Any(ByteSpan),
    Fun { span: ByteSpan, params: Vec<Type>, ret: Box<Type> }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeGuard {
    SubType(ByteSpan, Ident, Vec<Ident>),
    Constraint(ByteSpan, Ident, Type)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapPairType(ByteSpan, Type, Type);

#[derive(Debug, Clone, PartialEq)]
pub enum ConstantExpr {
    Literal(Literal),
    Nil(ByteSpan),
    List(ByteSpan, Vec<ConstantExpr>),
    Tuple(ByteSpan, Vec<ConstantExpr>),
    Map(ByteSpan, Vec<(ByteSpan, ConstantExpr, ConstantExpr)>),
    FunctionName(FunctionName),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Atom(Ident),
    String(Ident),
    Char(ByteSpan, char),
    Integer(ByteSpan, i64),
    BigInteger(ByteSpan, BigInt),
    Float(ByteSpan, f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    span: ByteSpan,
    name: Ident,
    fields: Vec<RecordField>
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordField {
    // name will only ever be Expr::Var or Expr::Literal(Literal::Atom)
    Static { span: ByteSpan, name: Expr, value: Option<Expr>, ty: Option<Type> },
    Dynamic { span: ByteSpan, name: Expr, value: Option<Expr>, ty: Option<Type> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordFieldPattern {
    // name will only ever be Expr::Var or Expr::Literal(Literal::Atom)
    Static { span: ByteSpan, name: Pattern, value: Option<Pattern> },
    Dynamic { span: ByteSpan, name: Pattern, value: Option<Pattern> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MapField {
    Assoc { span: ByteSpan, key: Expr, value: Expr },
    Exact { span: ByteSpan, key: Expr, value: Expr },
}

#[derive(Debug, Clone, PartialEq)]
pub enum MapFieldPattern {
    Assoc { span: ByteSpan, key: Pattern, value: Pattern },
    Exact { span: ByteSpan, key: Pattern, value: Pattern },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub span: ByteSpan,
    pub name: Ident,
    pub arity: usize,
    pub clauses: Vec<FunctionClause>
}
impl Function {
    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, clauses: Vec<FunctionClause>) -> Self {
        debug_assert!(clauses.len() > 0);
        let (head, rest) = clauses.split_first().expect("internal error: expected function to contain at least one clause");
        let name = head.name.clone().unwrap();
        let arity = head.params.len();
        // Check clauses
        for clause in rest.iter() {
            let clause_name = clause.name.clone().unwrap();
            let clause_arity = clause.params.len();
            if clause_name.name != name.name || clause_arity != arity {
                errs.push(to_lalrpop_err!(ParserError::UnexpectedFunctionClause {
                    found: FunctionName::from_clause(head),
                    expected: FunctionName::from_clause(clause)
                }));
                continue;
            }
        }
        Function { span, name, arity, clauses }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionClause {
    pub span: ByteSpan,
    pub name: Option<Ident>,
    pub params: Vec<Pattern>,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>
}
impl FunctionClause {
    pub fn new(span: ByteSpan, name: Option<Ident>, params: Vec<Pattern>, guard: Option<Vec<Guard>>, body: Vec<Expr>) -> Self {
        FunctionClause {
            span,
            name,
            params,
            guard,
            body
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Guard {
    pub span: ByteSpan,
    pub conditions: Vec<Expr>
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // An identifier/variable
    Var(Ident),
    Literal(Literal),
    // The various list forms
    Nil(ByteSpan),
    Cons(ByteSpan, Box<Expr>, Box<Expr>),
    // Other data structures
    Tuple(ByteSpan, Vec<Expr>),
    Map(ByteSpan, Vec<(ByteSpan, Expr, Expr)>),
    //Binary { span: ByteSpan, segments: Vec<BinarySegment> },
    // Accessing a record field value, e.g. Expr#myrec.field1
    RecordAccess { span: ByteSpan, lhs: Box<Expr>, record: Ident, field: Ident },
    // Referencing a record fields index, e.g. #myrec.field1
    RecordIndex { span: ByteSpan, record: Ident, field: Ident },
    // Record creation
    Record(Record),
    // A sequence of expressions, e.g. begin expr1, .., exprN end
    Begin(ByteSpan, Vec<Expr>),
    // Calls, e.g. foo(expr1, .., exprN)
    Apply { span: ByteSpan, lhs: Box<Expr>, args: Vec<Expr> },
    // Remote, e.g. Foo:Bar
    Remote { span: ByteSpan, module: Box<Expr>, function: Box<Expr> },
    // Comprehensions
    //Comprehension(Comprehension),
    // Binary/Unary expressions
    BinaryExpr { span: ByteSpan, lhs: Box<Expr>, op: BinaryOp, rhs: Box<Expr> },
    UnaryExpr { span: ByteSpan, op: UnaryOp, rhs: Box<Expr> },
    // Complex expressions
    Match(ByteSpan, Box<Expr>, Box<Expr>),
    If(ByteSpan, Vec<IfClause>),
    Catch(ByteSpan, Box<Expr>),
    Case(ByteSpan, Box<Expr>, Vec<Clause>),
    Receive { span: ByteSpan, clauses: Option<Vec<Clause>>, after: Option<Timeout> },
    Try { span: ByteSpan, exprs: Option<Vec<Expr>>, clauses: Option<Vec<Clause>>, catch_clauses: Option<Vec<TryClause>>, after: Option<Vec<Expr>> },
    //Fun(Fun),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryClause {
    pub span: ByteSpan,
    pub kind: Ident,
    pub error: Pattern,
    pub guard: Option<Vec<Guard>>,
    pub trace: Ident,
    pub body: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timeout(pub ByteSpan, pub Box<Expr>, pub Vec<Expr>);

#[derive(Debug, Clone, PartialEq)]
pub struct IfClause(pub ByteSpan, pub Vec<Expr>, pub Vec<Expr>);

#[derive(Debug, Clone, PartialEq)]
pub struct Clause{
    pub span: ByteSpan,
    pub pattern: Pattern,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Var(Ident),
    Literal(Literal),
    Nil(ByteSpan),
    Cons(ByteSpan, Box<Pattern>, Box<Pattern>),
    Tuple(ByteSpan, Vec<Expr>),
    Map(ByteSpan, Option<Box<Pattern>>, Vec<MapFieldPattern>),
    Record(ByteSpan, Ident, Vec<RecordFieldPattern>),
    RecordIndex(ByteSpan, Ident, Ident),
    Match(ByteSpan, Box<Pattern>, Box<Pattern>),
    BinaryExpr(ByteSpan, Box<Pattern>, BinaryOp, Box<Pattern>),
    UnaryExpr(ByteSpan, UnaryOp, Box<Pattern>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    // 100 !, right associative
    Send,
    // 150 orelse
    OrElse,
    // 160 andalso
    AndAlso,
    // 200 <all comparison operators>
    Equal, // right associative
    NotEqual,
    Lte,
    Lt,
    Gte,
    Gt,
    StrictEqual,
    StrictNotEqual,
    // 300 <all list operators>, right associative
    Append,
    Remove,
    // 400 <all add operators>, left associative
    Add,
    Sub,
    Bor,
    Bxor,
    Bsl,
    Bsr,
    Or,
    Xor,
    // 500 <all mul operators>, left associative
    Divide,
    Multiply,
    Div,
    Rem,
    Band,
    And,

    Error(ByteSpan)
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    // 600 <all prefix operators>
    Plus,
    Minus,
    Bnot,
    Not,

    Error(ByteSpan)
}

// Consumes a vector of ASTs that are flattened into a single AST. This is helpful if you want to
// merge ASTs from multiple files and be able to use the visitor pattern across all of them.
//pub fn flatten_asts(asts: Vec<AST>) -> AST {
    //asts.into_iter().flat_map(|ast| ast).collect()
//}
