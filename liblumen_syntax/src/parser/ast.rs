use std::fmt;
use std::cmp::Ordering;

use rug::Integer;

use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};

use crate::lexer::{Ident, Symbol};
use super::{ParseError, ParserError};

macro_rules! to_lalrpop_err (
    ($error:expr) => (lalrpop_util::ParseError::User { error: $error })
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Name {
    Atom(Ident),
    Var(Ident),
}
impl Name {
    pub fn symbol(&self) -> Symbol {
        match self {
            Name::Atom(Ident { ref name, .. }) => name.clone(),
            Name::Var(Ident { ref name, .. }) => name.clone(),
        }
    }
}
impl PartialOrd for Name {
    fn partial_cmp(&self, other: &Name) -> Option<Ordering> {
        self.symbol().partial_cmp(&other.symbol())
    }
}

#[derive(Debug, Clone)]
pub enum FunctionName {
    Resolved { span: ByteSpan, module: Option<Ident>, function: Ident, arity: Option<usize> },
    Unresolved { span: ByteSpan, module: Option<Name>, function: Name, arity: Option<usize> },
}
impl PartialEq for FunctionName {
    fn eq(&self, other: &FunctionName) -> bool {
        if self.module() != other.module() {
            return false;
        }
        if self.function() != other.function() {
            return false;
        }
        if self.arity() != other.arity() {
            return false;
        }

        true
    }
}
impl PartialOrd for FunctionName {
    fn partial_cmp(&self, other: &FunctionName) -> Option<Ordering> {
        let (xm, xf, xa) = (self.module(), self.function(), self.arity());
        let (ym, yf, ya) = (other.module(), other.function(), other.arity());
        match xm.partial_cmp(&ym) {
            None | Some(Ordering::Equal) =>
                match xf.partial_cmp(&yf) {
                    None | Some(Ordering::Equal) =>
                        xa.partial_cmp(&ya),
                    Some(order) =>
                        Some(order),
                },
            Some(order) =>
                Some(order),
        }
    }
}
impl FunctionName {
    pub fn span(&self) -> ByteSpan {
        match self {
            &FunctionName::Resolved { ref span, .. } => span.clone(),
            &FunctionName::Unresolved { ref span, .. } => span.clone(),
        }
    }

    pub fn module(&self) -> Option<Name> {
        match self {
            FunctionName::Resolved { module: Some(ref id), .. } =>
                Some(Name::Atom(id.clone())),
            FunctionName::Resolved { .. } =>
                None,
            FunctionName::Unresolved { ref module, .. } =>
                module.clone(),
        }
    }

    pub fn function(&self) -> Name {
        match self {
            FunctionName::Resolved { ref function, .. } =>
                Name::Atom(function.clone()),
            FunctionName::Unresolved { ref function, .. } =>
                function.clone()
        }
    }

    pub fn arity(&self) -> Option<usize> {
        match self {
            FunctionName::Resolved { ref arity, .. } => arity.clone(),
            FunctionName::Unresolved { ref arity, .. } => arity.clone(),
        }
    }

    pub fn detect(span: ByteSpan, module: Option<Name>, function: Name, arity: usize) -> Self {
        if module.is_none() {
            return match function {
                Name::Atom(f) =>
                    FunctionName::Resolved { span, module: None, function: f, arity: Some(arity) },
                Name::Var(_) =>
                    FunctionName::Unresolved { span, module: None, function, arity: Some(arity) },
            };
        }

        match module {
            Some(Name::Atom(m)) =>
                if let Name::Atom(f) = function {
                    return FunctionName::Resolved { span, module: Some(m), function: f, arity: Some(arity) };
                },
            _ => (),
        }

        FunctionName::Unresolved { span, module, function, arity: Some(arity) }
    }

    pub fn from_clause(clause: &FunctionClause) -> FunctionName {
        match clause {
            &FunctionClause::Named { ref name, ref span, ref params, .. } => {
                let arity = Some(params.len());
                FunctionName::Resolved {
                    span: span.clone(),
                    module: None,
                    function: name.clone(),
                    arity,
                }
            }
            &FunctionClause::Unnamed { .. } => {
                panic!("cannot create a FunctionName from an anonymous FunctionClause!")
            }
        }
    }
}
impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionName::Resolved { module: Some(ref m), ref function, arity: Some(a), .. } =>
                write!(f, "{}:{}/{}", m, function, a),
            FunctionName::Resolved { module: Some(ref m), ref function, .. } =>
                write!(f, "{}:{}", m, function),
            FunctionName::Resolved { ref function, arity: Some(a), .. } =>
                write!(f, "{}/{}", function, a),
            FunctionName::Resolved { ref function, .. } =>
                write!(f, "{}", function),
            FunctionName::Unresolved { module: Some(ref m), ref function, arity: Some(a), .. } =>
                write!(f, "{:?}:{:?}/{}", m, function, a),
            FunctionName::Unresolved { module: Some(ref m), ref function, .. } =>
                write!(f, "{:?}:{:?}", m, function),
            FunctionName::Unresolved { ref function, arity: Some(a), .. } =>
                write!(f, "{:?}/{}", function, a),
            FunctionName::Unresolved { ref function, .. } =>
                write!(f, "{:?}", function),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TopLevel {
    Attribute(Attribute),
    Record(Record),
    Function(Function)
}

#[derive(Debug, Clone)]
pub struct Module {
    pub span: ByteSpan,
    pub name: Ident,
    pub attributes: Vec<Attribute>,
    pub records: Vec<Record>,
    pub functions: Vec<Function>
}
impl PartialEq for Module {
    fn eq(&self, other: &Module) -> bool {
        if self.name.name != other.name.name {
            return false;
        }
        if self.attributes != other.attributes {
            return false;
        }
        if self.records != other.records {
            return false;
        }
        if self.functions != other.functions {
            return false;
        }
        true
    }
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

#[derive(Debug, Clone)]
pub enum Attribute {
    // Types
    Type(Type),
    Opaque(Type),
    ImportType { span: ByteSpan, module: Ident, types: Vec<FunctionName> },
    Spec { span: ByteSpan, name: Ident, sig: Vec<TypeSig> },
    // Builtins
    Export { span: ByteSpan, exports: Vec<FunctionName> },
    Import { span: ByteSpan, imports: Vec<FunctionName> },
    //Compile { span: ByteSpan, opts: Vec<Expr> },
    Vsn { span: ByteSpan, vsn: Expr },
    OnLoad { span: ByteSpan, fun: FunctionName },
    Callback { span: ByteSpan, module: Option<Ident>, fun: Ident, sig: Vec<TypeSig> },
    // User-defined attributes
    Custom { span: ByteSpan, name: Ident, value: Expr }
}
impl PartialEq for Attribute {
    fn eq(&self, other: &Attribute) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (&Attribute::Type(ref x), &Attribute::Type(ref y)) =>
                x == y,
            (&Attribute::Opaque(ref x), &Attribute::Opaque(ref y)) =>
                x == y,
            (&Attribute::ImportType { module: ref xm, types: ref xt, .. },
             &Attribute::ImportType { module: ref ym, types: ref yt, .. }) =>
                (xm == ym) && (xt == yt),
            (&Attribute::Spec { name: ref xname, sig: ref xsig, .. },
             &Attribute::Spec { name: ref yname, sig: ref ysig, .. }) =>
                (xname == yname) && (xsig == ysig),
            (&Attribute::Export { exports: ref x, .. },
             &Attribute::Export { exports: ref y, .. }) =>
                x == y,
            (&Attribute::Import { imports: ref x, .. },
             &Attribute::Import { imports: ref y, .. }) =>
                x == y,
            (&Attribute::Vsn { vsn: ref x, .. },
             &Attribute::Vsn { vsn: ref y, .. }) =>
                x == y,
            (&Attribute::OnLoad { fun: ref x, .. },
             &Attribute::OnLoad { fun: ref y, .. }) =>
                x == y,
            (&Attribute::Callback { module: ref xm, fun: ref xf, sig: ref xsig, .. },
             &Attribute::Callback { module: ref ym, fun: ref yf, sig: ref ysig, .. }) =>
                (xm == ym) && (xf == yf) && (xsig == ysig),
            (&Attribute::Custom { name: ref xname, value: ref xval, .. },
             &Attribute::Custom { name: ref yname, value: ref yval, .. }) =>
                (xname == yname) && (xval == yval),

            _ =>
                false
        }
    }
}
impl Attribute {
    pub fn new(span: ByteSpan, ident: Ident, values: Vec<Expr>) -> Result<Attribute, ParseError> {
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
                if let Expr::FunctionName(fun) = value {
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

#[derive(Debug, Clone)]
pub enum Type {
    Name(Name),
    Annotated { span: ByteSpan, name: Ident, ty: Box<Type> },
    Union { span: ByteSpan, lhs: Box<Type>, rhs: Box<Type> },
    Range { span: ByteSpan, start: Box<Type>, end: Box<Type> },
    BinaryOp { span: ByteSpan, lhs: Box<Type>, op: BinaryOp, rhs: Box<Type> },
    UnaryOp { span: ByteSpan, op: BinaryOp, rhs: Box<Type> },
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
impl PartialEq for Type {
    fn eq(&self, other: &Type) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (&Type::Annotated { name: ref x1, ty: ref x2, .. },
             &Type::Annotated { name: ref y1, ty: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::Union { lhs: ref x1, rhs: ref x2, .. },
             &Type::Union { lhs: ref y1, rhs: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::Range { start: ref x1, end: ref x2, .. },
             &Type::Range { start: ref y1, end: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::BinaryOp { lhs: ref x1, op: ref x2, rhs: ref x3, .. },
             &Type::BinaryOp { lhs: ref y1, op: ref y2, rhs: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Type::UnaryOp { op: ref x1, rhs: ref x2, .. },
             &Type::UnaryOp { op: ref y1, rhs: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::Generic { name: ref x1, params: ref x2, ty: ref x3, .. },
             &Type::Generic { name: ref y1, params: ref y2, ty: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Type::Remote { module: ref x1, fun: ref x2, args: ref x3, .. },
             &Type::Remote { module: ref y1, fun: ref y2, args: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Type::Nil(_), &Type::Nil(_)) =>
                true,
            (&Type::List(_, ref x), &Type::List(_, ref y)) =>
                x == y,
            (&Type::NonEmptyList(_, ref x), &Type::List(_, ref y)) =>
                x == y,
            (&Type::Map(_, ref x), &Type::Map(_, ref y)) =>
                x == y,
            (&Type::Tuple(_, ref x), &Type::Tuple(_, ref y)) =>
                x == y,
            (&Type::Record { name: ref x1, fields: ref x2, .. },
             &Type::Record { name: ref y1, fields: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::Binary { head: ref x1, tail: ref x2, .. },
             &Type::Binary { head: ref y1, tail: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Type::Integer(_, x), &Type::Integer(_, y)) =>
                x == y,
            (&Type::Char(_, x), &Type::Char(_, y)) =>
                x == y,
            (&Type::Fun(ref x), &Type::Fun(ref y)) =>
                x == y,
            _ =>
                false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeSig {
    sig: FunType,
    constraints: Vec<TypeGuard>
}

#[derive(Debug, Clone)]
pub enum FunType {
    Any(ByteSpan),
    Fun { span: ByteSpan, params: Vec<Type>, ret: Box<Type> }
}
impl PartialEq for FunType {
    fn eq(&self, other: &FunType) -> bool {
        match (self, other) {
            (&FunType::Any(_), &FunType::Any(_)) =>
                true,
            (&FunType::Fun { params: ref x1, ret: ref x2, .. },
             &FunType::Fun { params: ref y1, ret: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            _ =>
                false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeGuard {
    SubType(ByteSpan, Ident, Vec<Ident>),
    Constraint(ByteSpan, Ident, Type)
}
impl PartialEq for TypeGuard {
    fn eq(&self, other: &TypeGuard) -> bool {
        match (self, other) {
            (&TypeGuard::SubType(_, ref x1, ref x2), &TypeGuard::SubType(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&TypeGuard::Constraint(_, ref x1, ref x2), &TypeGuard::Constraint(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            _ =>
                false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapPairType(ByteSpan, Type, Type);
impl PartialEq for MapPairType {
    fn eq(&self, other: &MapPairType) -> bool {
        (self.1 == other.1) && (self.2 == other.2)
    }
}

#[derive(Debug, Clone)]
pub enum Literal {
    Atom(Ident),
    String(Ident),
    Char(ByteSpan, char),
    Integer(ByteSpan, i64),
    BigInteger(ByteSpan, Integer),
    Float(ByteSpan, f64),
}
impl PartialEq for Literal {
    fn eq(&self, other: &Literal) -> bool {
        match (self, other) {
            (&Literal::Atom(Ident { name: x, .. }), &Literal::Atom(Ident { name: y, .. })) =>
                x == y,
            (&Literal::Atom(_), _) =>
                false,
            (_, &Literal::Atom(_)) =>
                false,
            (&Literal::String(Ident { name: x, .. }), &Literal::String(Ident { name: y, .. })) =>
                x == y,
            (&Literal::String(_), _) =>
                false,
            (_, &Literal::String(_)) =>
                false,
            (x, y) =>
                x.partial_cmp(y) == Some(Ordering::Equal),
        }
    }
}
impl PartialOrd for Literal {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn partial_cmp(&self, other: &Literal) -> Option<Ordering> {
        match (self, other) {
            (&Literal::String(Ident { name: ref x, .. }), &Literal::String(Ident { name: ref y, .. })) =>
                x.partial_cmp(y),
            (&Literal::String(_), _) =>
                Some(Ordering::Greater),
            (_, &Literal::String(_)) =>
                Some(Ordering::Less),
            (&Literal::Atom(Ident { name: ref x, .. }), &Literal::Atom(Ident { name: ref y, .. })) =>
                x.partial_cmp(y),
            (&Literal::Atom(_), _) =>
                Some(Ordering::Greater),
            (_, &Literal::Atom(_)) =>
                Some(Ordering::Less),
            (&Literal::Integer(_, x), &Literal::Integer(_, y)) =>
                x.partial_cmp(&y),
            (&Literal::Integer(_, x), &Literal::BigInteger(_, ref y)) =>
                x.partial_cmp(y),
            (&Literal::Integer(_, x), &Literal::Float(_, y)) =>
                (x as f64).partial_cmp(&y),
            (&Literal::Integer(_, x), &Literal::Char(_, y)) =>
                x.partial_cmp(&(y as i64)),
            (&Literal::BigInteger(_, ref x), &Literal::BigInteger(_, ref y)) =>
                x.partial_cmp(y),
            (&Literal::BigInteger(_, ref x), &Literal::Integer(_, y)) =>
                x.partial_cmp(&y),
            (&Literal::BigInteger(_, ref x), &Literal::Float(_, y)) =>
                x.partial_cmp(&y),
            (&Literal::BigInteger(_, ref x), &Literal::Char(_, y)) =>
                x.partial_cmp(&(y as i64)),
            (&Literal::Float(_, x), &Literal::Float(_, y)) =>
                x.partial_cmp(&y),
            (&Literal::Float(_, x), &Literal::Integer(_, y)) =>
                x.partial_cmp(&(y as f64)),
            (&Literal::Float(_, x), &Literal::BigInteger(_, ref y)) =>
                x.partial_cmp(y),
            (&Literal::Float(_, x), &Literal::Char(_, y)) =>
                x.partial_cmp(&((y as i64) as f64)),
            (&Literal::Char(_, x), &Literal::Char(_, y)) =>
                x.partial_cmp(&y),
            (&Literal::Char(_, x), &Literal::Integer(_, y)) =>
                (x as i64).partial_cmp(&y),
            (&Literal::Char(_, x), &Literal::BigInteger(_, ref y)) =>
                (x as i64).partial_cmp(y),
            (&Literal::Char(_, x), &Literal::Float(_, y)) =>
                ((x as i64) as f64).partial_cmp(&y),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub span: ByteSpan,
    pub name: Ident,
    pub fields: Vec<RecordField>
}
impl PartialEq for Record {
    fn eq(&self, other: &Record) -> bool {
        (self.name == other.name) && (self.fields == other.fields)
    }
}

#[derive(Debug, Clone)]
pub struct RecordField {
    pub span: ByteSpan,
    pub name: Name,
    pub value: Option<Expr>,
    pub ty: Option<Type>,
}
impl PartialEq for RecordField {
    fn eq(&self, other: &RecordField) -> bool {
        (self.name == other.name) && (self.value == other.value) && (self.ty == other.ty)
    }
}

#[derive(Debug, Clone)]
pub struct RecordFieldPattern {
    pub span: ByteSpan,
    pub name: Name,
    pub value: Option<Pattern>,
}
impl PartialEq for RecordFieldPattern {
    fn eq(&self, other: &RecordFieldPattern) -> bool {
        (self.name == other.name) && (self.value == other.value)
    }
}

#[derive(Debug, Clone)]
pub enum MapField {
    Assoc { span: ByteSpan, key: Expr, value: Expr },
    Exact { span: ByteSpan, key: Expr, value: Expr },
}
impl PartialEq for MapField {
    fn eq(&self, other: &MapField) -> bool {
        (self.key() == other.key()) && (self.value() == other.value())
    }
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
impl PartialOrd for MapField {
    fn partial_cmp(&self, other: &MapField) -> Option<Ordering> {
        match self.key().partial_cmp(&other.key()) {
            None =>
                None,
            Some(Ordering::Equal) =>
                self.value().partial_cmp(&other.value()),
            Some(order) =>
                Some(order)
        }
    }
}

#[derive(Debug, Clone)]
pub enum MapFieldPattern {
    Assoc { span: ByteSpan, key: Pattern, value: Pattern },
    Exact { span: ByteSpan, key: Pattern, value: Pattern },
}
impl PartialEq for MapFieldPattern {
    fn eq(&self, other: &MapFieldPattern) -> bool {
        (self.key() == other.key()) && (self.value() == other.value())
    }
}
impl MapFieldPattern {
    pub fn key(&self) -> Pattern {
        match self {
            &MapFieldPattern::Assoc { ref key, .. } => key.clone(),
            &MapFieldPattern::Exact { ref key, .. } => key.clone(),
        }
    }

    pub fn value(&self) -> Pattern {
        match self {
            &MapFieldPattern::Assoc { ref value, .. } => value.clone(),
            &MapFieldPattern::Exact { ref value, .. } => value.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Function {
    Named { span: ByteSpan, name: Ident, arity: usize, clauses: Vec<FunctionClause> },
    Unnamed { span: ByteSpan, arity: usize, clauses: Vec<FunctionClause> },
}
impl PartialEq for Function {
    fn eq(&self, other: &Function) -> bool {
        match (self, other) {
            (&Function::Named { name: ref x1, arity: ref x2, clauses: ref x3, .. },
             &Function::Named { name: ref y1, arity: ref y2, clauses: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Function::Unnamed { arity: ref x1, clauses: ref x2, .. },
             &Function::Unnamed { arity: ref y1, clauses: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            _ =>
                false,
        }
    }
}
impl Function {
    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, clauses: Vec<FunctionClause>) -> Self {
        debug_assert!(clauses.len() > 0);
        let (head, rest) = clauses.split_first().expect("internal error: expected function to contain at least one clause");
        match head {
            FunctionClause::Named { ref name, ref params, .. } => {
                let arity = params.len();
                // Check clauses
                for clause in rest.iter() {
                    match clause {
                        FunctionClause::Named { name: ref clause_name, params: ref clause_params, .. } => {
                            let clause_arity = clause_params.len();
                            if clause_name.name != name.name || clause_arity != arity {
                                errs.push(to_lalrpop_err!(ParserError::UnexpectedFunctionClause {
                                    found: FunctionName::from_clause(head),
                                    expected: FunctionName::from_clause(clause)
                                }));
                                continue;
                            }
                        }
                        FunctionClause::Unnamed { .. } => {
                            errs.push(to_lalrpop_err!(ParserError::UnexpectedFunctionClause {
                                found: FunctionName::from_clause(head),
                                expected: FunctionName::from_clause(clause)
                            }));
                            continue;
                        }
                    }
                }
                Function::Named { span, name: name.clone(), arity, clauses }
            }
            FunctionClause::Unnamed { ref params, .. } => {
                let arity = params.len();
                // Check clauses
                for clause in rest.iter() {
                    match clause {
                        FunctionClause::Unnamed { params: ref clause_params, .. } => {
                            let clause_arity = clause_params.len();
                            if clause_arity != arity {
                                errs.push(to_lalrpop_err!(ParserError::MismatchedFunctionClause {
                                    found: head.clone(),
                                    expected: clause.clone(),
                                }));
                                continue;
                            }
                        }
                        FunctionClause::Named { .. } => {
                            errs.push(to_lalrpop_err!(ParserError::MismatchedFunctionClause {
                                found: head.clone(),
                                expected: clause.clone(),
                            }));
                            continue;
                        }
                    }
                }
                Function::Unnamed { span, arity, clauses }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum FunctionClause {
    Named { span: ByteSpan, name: Ident, params: Vec<Pattern>, guard: Option<Vec<Guard>>, body: Vec<Expr> },
    Unnamed { span: ByteSpan, params: Vec<Pattern>, guard: Option<Vec<Guard>>, body: Vec<Expr> }
}
impl PartialEq for FunctionClause {
    fn eq(&self, other: &FunctionClause) -> bool {
        match (self, other) {
            (&FunctionClause::Named { name: ref x1, params: ref x2, guard: ref x3, body: ref x4, .. },
             &FunctionClause::Named { name: ref y1, params: ref y2, guard: ref y3, body: ref y4, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3) && (x4 == y4),
            (&FunctionClause::Unnamed { params: ref x1, guard: ref x2, body: ref x3, .. },
             &FunctionClause::Unnamed { params: ref y1, guard: ref y2, body: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            _ =>
                false,
        }
    }
}
impl FunctionClause {
    pub fn new(span: ByteSpan, name: Option<Ident>, params: Vec<Pattern>, guard: Option<Vec<Guard>>, body: Vec<Expr>) -> Self {
        match name {
            None =>
                FunctionClause::Unnamed {
                    span,
                    params,
                    guard,
                    body
                },
            Some(name) =>
                FunctionClause::Named {
                    span,
                    name,
                    params,
                    guard,
                    body
                },
        }
    }

    pub fn span(&self) -> ByteSpan {
        match *self {
            FunctionClause::Named { ref span, .. } => span.clone(),
            FunctionClause::Unnamed { ref span, .. } => span.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Guard {
    pub span: ByteSpan,
    pub conditions: Vec<Expr>
}
impl PartialEq for Guard {
    fn eq(&self, other: &Guard) -> bool {
        self.conditions == other.conditions
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    // An identifier/variable
    Var(Ident),
    Literal(Literal),
    FunctionName(FunctionName),
    // The various list forms
    Nil(ByteSpan),
    Cons(ByteSpan, Box<Expr>, Box<Expr>),
    // Other data structures
    Tuple(ByteSpan, Vec<Expr>),
    Map(ByteSpan, Option<Box<Expr>>, Vec<MapField>),
    Binary(ByteSpan, Vec<BinaryElement>),
    // Accessing a record field value, e.g. Expr#myrec.field1
    RecordAccess(ByteSpan, Box<Expr>, Ident, Ident),
    // Referencing a record fields index, e.g. #myrec.field1
    RecordIndex(ByteSpan, Ident, Ident),
    // Update a record field value, e.g. Expr#myrec.field1
    RecordUpdate(ByteSpan, Box<Expr>, Ident, Vec<RecordField>),
    // Record creation
    Record(ByteSpan, Ident, Vec<RecordField>),
    // A sequence of expressions, e.g. begin expr1, .., exprN end
    Begin(ByteSpan, Vec<Expr>),
    // Calls, e.g. foo(expr1, .., exprN)
    Apply { span: ByteSpan, lhs: Box<Expr>, args: Vec<Expr> },
    // Remote, e.g. Foo:Bar
    Remote { span: ByteSpan, module: Box<Expr>, function: Box<Expr> },
    // Comprehensions
    ListComprehension(ByteSpan, Box<Expr>, Vec<Expr>),
    BinaryComprehension(ByteSpan, Box<Expr>, Vec<Expr>),
    Generator(ByteSpan, Box<Expr>, Box<Expr>),
    BinaryGenerator(ByteSpan, Box<Expr>, Box<Expr>),
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
    Fun(Function),
}
impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        match (self, other) {
            (&Expr::Var(ref x), &Expr::Var(ref y)) =>
                x.eq(y),
            (&Expr::Var(_), _) =>
                false,
            (&Expr::Literal(ref x), &Expr::Literal(ref y)) =>
                x.eq(y),
            (&Expr::Literal(_), _) =>
                false,
            (&Expr::Nil(_), &Expr::Nil(_)) =>
                true,
            (&Expr::Nil(_), _) =>
                false,
            (&Expr::Cons(_, ref xlhs, ref xrhs), &Expr::Cons(_, ref ylhs, ref yrhs)) =>
                xlhs.eq(ylhs) && xrhs.eq(yrhs),
            (&Expr::Cons(_, _, _), _) =>
                false,
            (&Expr::Tuple(_, ref x), &Expr::Tuple(_, ref y)) =>
                x.eq(y),
            (&Expr::Tuple(_, _), _) =>
                false,
            (&Expr::Map(_, Some(ref xlhs), ref xfields), &Expr::Map(_, Some(ref ylhs), ref yfields)) =>
                xlhs.eq(ylhs) && xfields.eq(yfields),
            (&Expr::Map(_, None, ref xfields), &Expr::Map(_, None, ref yfields)) =>
                xfields.eq(yfields),
            (&Expr::Binary(_, ref x), &Expr::Binary(_, ref y)) =>
                x.eq(y),
            (&Expr::FunctionName(ref x), &Expr::FunctionName(ref y)) =>
                x.eq(y),
            (&Expr::RecordAccess(_, ref x1, ref x2, ref x3), &Expr::RecordAccess(_, ref y1, ref y2, ref y3)) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Expr::RecordIndex(_, ref x1, ref x2), &Expr::RecordIndex(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::RecordUpdate(_, ref x1, ref x2, ref x3), &Expr::RecordUpdate(_, ref y1, ref y2, ref y3)) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Expr::Record(_, ref x1, ref x2), &Expr::Record(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Begin(_, ref x1), &Expr::Begin(_, ref y1)) =>
                x1 == y1,
            (&Expr::Apply { lhs: ref x1, args: ref x2, .. }, &Expr::Apply { lhs: ref y1, args: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Remote { module: ref x1, function: ref x2, .. }, &Expr::Remote { module: ref y1, function: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::ListComprehension(_, ref x1, ref x2), &Expr::ListComprehension(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::BinaryComprehension(_, ref x1, ref x2), &Expr::BinaryComprehension(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Generator(_, ref x1, ref x2), &Expr::Generator(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::BinaryGenerator(_, ref x1, ref x2), &Expr::BinaryGenerator(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::BinaryExpr { lhs: ref x1, op: ref x2, rhs: ref x3, .. },
             &Expr::BinaryExpr { lhs: ref y1, op: ref y2, rhs: ref y3, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Expr::UnaryExpr { op: ref x1, rhs: ref x2, .. },
             &Expr::UnaryExpr { op: ref y1, rhs: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Match(_, ref x1, ref x2), &Expr::Match(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::If(_, ref x1), &Expr::If(_, ref y1)) =>
                x1 == y1,
            (&Expr::Catch(_, ref x1), &Expr::Catch(_, ref y1)) =>
                x1 == y1,
            (&Expr::Case(_, ref x1, ref x2), &Expr::Case(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Receive { clauses: ref x1, after: ref x2, ..}, &Expr::Receive { clauses: ref y1, after: ref y2, .. }) =>
                (x1 == y1) && (x2 == y2),
            (&Expr::Try { exprs: ref x1, clauses: ref x2, catch_clauses: ref x3, after: ref x4, .. },
             &Expr::Try { exprs: ref y1, clauses: ref y2, catch_clauses: ref y3, after: ref y4, .. }) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3) && (x4 == y4),
            (&Expr::Fun(ref x1), &Expr::Fun(ref y1)) =>
                x1 == y1,
            _ =>
                false
        }
    }
}
impl PartialOrd for Expr {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn partial_cmp(&self, other: &Expr) -> Option<Ordering> {
        match (self, other) {
            (&Expr::Binary(_, _), &Expr::Binary(_, _)) =>
                None,
            (&Expr::Binary(_, _), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Binary(_, _)) =>
                Some(Ordering::Less),
            (&Expr::Cons(_, ref xlhs, ref xrhs), &Expr::Cons(_, ref ylhs, ref yrhs)) =>
                match xlhs.partial_cmp(ylhs) {
                    None => xrhs.partial_cmp(yrhs),
                    Some(order) => Some(order),
                },
            (&Expr::Cons(_, _, _), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Cons(_, _, _)) =>
                Some(Ordering::Less),
            (&Expr::Nil(_), &Expr::Nil(_)) =>
                Some(Ordering::Equal),
            (&Expr::Nil(_), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Nil(_)) =>
                Some(Ordering::Less),
            (&Expr::Map(_, Some(ref xlhs), ref xfields), &Expr::Map(_, Some(ref ylhs), ref yfields)) =>
                match xlhs.partial_cmp(ylhs) {
                    None => xfields.partial_cmp(yfields),
                    Some(order) => Some(order),
                },
            (&Expr::Map(_, None, ref xfields), &Expr::Map(_, None, ref yfields)) =>
                xfields.partial_cmp(yfields),
            (&Expr::Map(_, _, _), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Map(_, _, _)) =>
                Some(Ordering::Less),
            (&Expr::Tuple(_, ref x), &Expr::Tuple(_, ref y)) =>
                x.partial_cmp(y),
            (&Expr::Tuple(_, _), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Tuple(_, _)) =>
                Some(Ordering::Less),
            (&Expr::Fun(_), &Expr::Fun(_)) =>
                None,
            (&Expr::Fun(_), _) =>
                Some(Ordering::Greater),
            (_, &Expr::Fun(_)) =>
                Some(Ordering::Less),
            (&Expr::Literal(ref x), &Expr::Literal(ref y)) =>
                x.partial_cmp(y),
            _ =>
                None
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryElement {
    pub span: ByteSpan,
    pub bit_expr: Expr,
    pub bit_size: Option<Expr>,
    pub bit_type: Option<Vec<BitType>>,
}
impl PartialEq for BinaryElement {
    fn eq(&self, other: &BinaryElement) -> bool {
        (self.bit_expr == other.bit_expr) &&
        (self.bit_size == other.bit_size) &&
        (self.bit_type == other.bit_type)
    }
}

#[derive(Debug, Clone)]
pub enum BitType {
    Name(ByteSpan, Ident),
    Sized(ByteSpan, Ident, i64),
}
impl PartialEq for BitType {
    fn eq(&self, other: &BitType) -> bool {
        match (self, other) {
            (&BitType::Name(_, ref x1), &BitType::Name(_, ref y1)) =>
                x1 == y1,
            (&BitType::Sized(_, ref x1, ref x2), &BitType::Sized(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            _ =>
                false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TryClause {
    pub span: ByteSpan,
    pub kind: Name,
    pub error: Pattern,
    pub guard: Option<Vec<Guard>>,
    pub trace: Ident,
    pub body: Vec<Expr>,
}
impl PartialEq for TryClause {
    fn eq(&self, other: &TryClause) -> bool {
        (self.kind == other.kind) &&
        (self.error == other.error) &&
        (self.guard == other.guard) &&
        (self.trace == other.trace) &&
        (self.body == other.body)
    }
}

#[derive(Debug, Clone)]
pub struct Timeout(pub ByteSpan, pub Box<Expr>, pub Vec<Expr>);
impl PartialEq for Timeout {
    fn eq(&self, other: &Timeout) -> bool {
        (self.1 == other.1) && (self.2 == other.2)
    }
}

#[derive(Debug, Clone)]
pub struct IfClause(pub ByteSpan, pub Vec<Expr>, pub Vec<Expr>);
impl PartialEq for IfClause {
    fn eq(&self, other: &IfClause) -> bool {
        (self.1 == other.1) && (self.2 == other.2)
    }
}

#[derive(Debug, Clone)]
pub struct Clause{
    pub span: ByteSpan,
    pub pattern: Pattern,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}
impl PartialEq for Clause {
    fn eq(&self, other: &Clause) -> bool {
        (self.pattern == other.pattern) &&
        (self.guard == other.guard) &&
        (self.body == other.body)
    }
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Var(Ident),
    Literal(Literal),
    Nil(ByteSpan),
    Cons(ByteSpan, Box<Pattern>, Box<Pattern>),
    Tuple(ByteSpan, Vec<Expr>),
    Binary(ByteSpan, Vec<BinaryElement>),
    Map(ByteSpan, Option<Box<Pattern>>, Vec<MapFieldPattern>),
    Record(ByteSpan, Ident, Vec<RecordFieldPattern>),
    RecordIndex(ByteSpan, Ident, Ident),
    Match(ByteSpan, Box<Pattern>, Box<Pattern>),
    BinaryExpr(ByteSpan, Box<Pattern>, BinaryOp, Box<Pattern>),
    UnaryExpr(ByteSpan, UnaryOp, Box<Pattern>),
}
impl PartialEq for Pattern {
    fn eq(&self, other: &Pattern) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (&Pattern::Var(ref x), &Pattern::Var(ref y)) =>
                x == y,
            (&Pattern::Literal(ref x), &Pattern::Literal(ref y)) =>
                x == y,
            (&Pattern::Nil(_), &Pattern::Nil(_)) =>
                true,
            (&Pattern::Cons(_, ref x1, ref x2), &Pattern::Cons(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Pattern::Tuple(_, ref x), &Pattern::Tuple(_, ref y)) =>
                x == y,
            (&Pattern::Binary(_, ref x), &Pattern::Binary(_, ref y)) =>
                x == y,
            (&Pattern::Map(_, ref x1, ref x2), &Pattern::Map(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Pattern::Record(_, ref x1, ref x2), &Pattern::Record(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Pattern::RecordIndex(_, ref x1, ref x2), &Pattern::RecordIndex(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Pattern::Match(_, ref x1, ref x2), &Pattern::Match(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            (&Pattern::BinaryExpr(_, ref x1, ref x2, ref x3), &Pattern::BinaryExpr(_, ref y1, ref y2, ref y3)) =>
                (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Pattern::UnaryExpr(_, ref x1, ref x2), &Pattern::UnaryExpr(_, ref y1, ref y2)) =>
                (x1 == y1) && (x2 == y2),
            _ =>
                false,
        }
    }
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    // 600 <all prefix operators>
    Plus,
    Minus,
    Bnot,
    Not,
}
