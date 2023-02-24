///! These are expression types that are used internally during translation from AST to CST
///!
///! The final CST never contains these expression types.
use std::fmt;

use firefly_binary::BinaryEntrySpecifier;
use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_intern::{symbols, Ident, Symbol};
use firefly_syntax_base::*;

#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub enum IExpr {
    Alias(IAlias),
    Apply(IApply),
    Binary(IBinary),
    Call(ICall),
    Case(ICase),
    Catch(ICatch),
    Cons(ICons),
    Exprs(IExprs),
    Fun(IFun),
    If(IIf),
    LetRec(ILetRec),
    Literal(Literal),
    Match(IMatch),
    Map(IMap),
    PrimOp(IPrimOp),
    Protect(IProtect),
    Receive1(IReceive1),
    Receive2(IReceive2),
    Set(ISet),
    Simple(ISimple),
    Tuple(ITuple),
    Try(ITry),
    Var(Var),
}
impl fmt::Display for IExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut pp = crate::printer::PrettyPrinter::new(f);
        pp.print_iexpr(self)
    }
}
impl Annotated for IExpr {
    fn annotations(&self) -> &Annotations {
        match self {
            Self::Alias(expr) => expr.annotations(),
            Self::Apply(expr) => expr.annotations(),
            Self::Binary(expr) => expr.annotations(),
            Self::Call(expr) => expr.annotations(),
            Self::Case(expr) => expr.annotations(),
            Self::Catch(expr) => expr.annotations(),
            Self::Cons(expr) => expr.annotations(),
            Self::Exprs(expr) => expr.annotations(),
            Self::Fun(expr) => expr.annotations(),
            Self::If(expr) => expr.annotations(),
            Self::LetRec(expr) => expr.annotations(),
            Self::Literal(expr) => expr.annotations(),
            Self::Match(expr) => expr.annotations(),
            Self::Map(expr) => expr.annotations(),
            Self::PrimOp(expr) => expr.annotations(),
            Self::Protect(expr) => expr.annotations(),
            Self::Receive1(expr) => expr.annotations(),
            Self::Receive2(expr) => expr.annotations(),
            Self::Set(expr) => expr.annotations(),
            Self::Simple(expr) => expr.annotations(),
            Self::Try(expr) => expr.annotations(),
            Self::Tuple(expr) => expr.annotations(),
            Self::Var(expr) => expr.annotations(),
        }
    }

    fn annotations_mut(&mut self) -> &mut Annotations {
        match self {
            Self::Alias(expr) => expr.annotations_mut(),
            Self::Apply(expr) => expr.annotations_mut(),
            Self::Binary(expr) => expr.annotations_mut(),
            Self::Call(expr) => expr.annotations_mut(),
            Self::Case(expr) => expr.annotations_mut(),
            Self::Catch(expr) => expr.annotations_mut(),
            Self::Cons(expr) => expr.annotations_mut(),
            Self::Exprs(expr) => expr.annotations_mut(),
            Self::Fun(expr) => expr.annotations_mut(),
            Self::If(expr) => expr.annotations_mut(),
            Self::LetRec(expr) => expr.annotations_mut(),
            Self::Literal(expr) => expr.annotations_mut(),
            Self::Match(expr) => expr.annotations_mut(),
            Self::Map(expr) => expr.annotations_mut(),
            Self::PrimOp(expr) => expr.annotations_mut(),
            Self::Protect(expr) => expr.annotations_mut(),
            Self::Receive1(expr) => expr.annotations_mut(),
            Self::Receive2(expr) => expr.annotations_mut(),
            Self::Set(expr) => expr.annotations_mut(),
            Self::Simple(expr) => expr.annotations_mut(),
            Self::Try(expr) => expr.annotations_mut(),
            Self::Tuple(expr) => expr.annotations_mut(),
            Self::Var(expr) => expr.annotations_mut(),
        }
    }
}
impl IExpr {
    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Literal(_) | Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_safe(&self) -> bool {
        match self {
            Self::Cons(_) | Self::Tuple(_) | Self::Literal(_) => true,
            // Fun
            Self::Var(v) if v.arity.is_some() => false,
            // Ordinary variable
            Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_simple(&self) -> bool {
        match self {
            Self::Var(_) | Self::Literal(_) => true,
            Self::Cons(ICons { head, tail, .. }) => head.is_simple() && tail.is_simple(),
            Self::Tuple(ITuple { elements, .. }) => elements.iter().all(|e| e.is_simple()),
            Self::Map(IMap { pairs, .. }) => pairs
                .iter()
                .all(|pair| pair.key.iter().all(|k| k.is_simple()) && pair.value.is_simple()),
            _ => false,
        }
    }

    pub fn is_data(&self) -> bool {
        match self {
            Self::Literal(_) | Self::Cons(_) | Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_literal(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            _ => false,
        }
    }

    pub fn is_boolean(&self) -> bool {
        match self {
            Self::Literal(Literal {
                value: Lit::Atom(a),
                ..
            }) => a.is_boolean(),
            _ => false,
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Self::Literal(Literal {
                value: Lit::Atom(a),
                ..
            }) if a.is_boolean() => Some(*a == symbols::True),
            _ => None,
        }
    }

    pub fn is_atom(&self) -> bool {
        match self {
            Self::Literal(Literal {
                value: Lit::Atom(_),
                ..
            }) => true,
            _ => false,
        }
    }

    pub fn is_atom_value(&self, symbol: Symbol) -> bool {
        match self {
            Self::Literal(Literal {
                value: Lit::Atom(a),
                ..
            }) => *a == symbol,
            _ => false,
        }
    }

    pub fn as_atom(&self) -> Option<Symbol> {
        match self {
            Self::Literal(Literal {
                value: Lit::Atom(a),
                ..
            }) => Some(*a),
            _ => None,
        }
    }

    pub fn coerce_to_float(self) -> Self {
        match self {
            Self::Literal(Literal {
                span,
                annotations,
                value: Lit::Integer(i),
            }) => match i.to_efloat() {
                Ok(float) => Self::Literal(Literal {
                    span,
                    annotations,
                    value: Lit::Float(float),
                }),
                Err(_) => Self::Literal(Literal {
                    span,
                    annotations,
                    value: Lit::Integer(i),
                }),
            },
            other => other,
        }
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IAlias {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub var: Var,
    pub pattern: Box<IExpr>,
}
annotated!(IAlias);
impl IAlias {
    pub fn new(span: SourceSpan, var: Var, pattern: IExpr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            var,
            pattern: Box::new(pattern),
        }
    }
}
impl Eq for IAlias {}
impl PartialEq for IAlias {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.pattern == other.pattern
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IApply {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub callee: Vec<IExpr>,
    pub args: Vec<IExpr>,
}
annotated!(IApply);
impl IApply {
    pub fn new(span: SourceSpan, callee: IExpr, args: Vec<IExpr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            callee: vec![callee],
            args,
        }
    }
}
impl Eq for IApply {}
impl PartialEq for IApply {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IBinary {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub segments: Vec<IBitstring>,
}
annotated!(IBinary);
impl IBinary {
    pub fn new(span: SourceSpan, segments: Vec<IBitstring>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            segments,
        }
    }
}
impl Eq for IBinary {}
impl PartialEq for IBinary {
    fn eq(&self, other: &Self) -> bool {
        self.segments == other.segments
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IBitstring {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub value: Box<IExpr>,
    pub size: Vec<IExpr>,
    pub spec: BinaryEntrySpecifier,
}
annotated!(IBitstring);
impl Eq for IBitstring {}
impl PartialEq for IBitstring {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.size == other.size && self.spec == other.spec
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ICall {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub module: Box<IExpr>,
    pub function: Box<IExpr>,
    pub args: Vec<IExpr>,
}
annotated!(ICall);
impl ICall {
    pub fn new(span: SourceSpan, module: Symbol, function: Symbol, args: Vec<IExpr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            module: Box::new(IExpr::Literal(Literal::atom(span, module))),
            function: Box::new(IExpr::Literal(Literal::atom(span, function))),
            args,
        }
    }

    pub fn is_static(&self, module: Symbol, function: Symbol, arity: usize) -> bool {
        if self.args.len() != arity {
            return false;
        }

        self.module.is_atom_value(module) && self.function.is_atom_value(function)
    }
}
impl Eq for ICall {}
impl PartialEq for ICall {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ICase {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub args: Vec<IExpr>,
    pub clauses: Vec<IClause>,
    pub fail: Box<IClause>,
}
annotated!(ICase);
impl Eq for ICase {}
impl PartialEq for ICase {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args && self.clauses == other.clauses && self.fail == other.fail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ICatch {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub body: Vec<IExpr>,
}
annotated!(ICatch);
impl Eq for ICatch {}
impl PartialEq for ICatch {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IClause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub patterns: Vec<IExpr>,
    pub guards: Vec<IExpr>,
    pub body: Vec<IExpr>,
}
annotated!(IClause);
impl IClause {
    pub fn new(
        span: SourceSpan,
        patterns: Vec<IExpr>,
        guards: Vec<IExpr>,
        body: Vec<IExpr>,
    ) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            patterns,
            guards,
            body,
        }
    }
}
impl Eq for IClause {}
impl PartialEq for IClause {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns && self.guards == other.guards && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ICons {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub head: Box<IExpr>,
    pub tail: Box<IExpr>,
}
annotated!(ICons);
impl ICons {
    pub fn new(span: SourceSpan, head: IExpr, tail: IExpr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }
}
impl Eq for ICons {}
impl PartialEq for ICons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IExprs {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub bodies: Vec<Vec<IExpr>>,
}
annotated!(IExprs);
impl IExprs {
    pub fn new(span: SourceSpan, bodies: Vec<Vec<IExpr>>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            bodies,
        }
    }
}
impl Eq for IExprs {}
impl PartialEq for IExprs {
    fn eq(&self, other: &Self) -> bool {
        self.bodies == other.bodies
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IFun {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub id: Option<Ident>,
    pub name: Option<Ident>,
    pub vars: Vec<Var>,
    pub clauses: Vec<IClause>,
    pub fail: Box<IClause>,
}
impl fmt::Display for IFun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut pp = crate::printer::PrettyPrinter::new(f);
        pp.print_ifun(self)
    }
}
annotated!(IFun);
impl Eq for IFun {}
impl PartialEq for IFun {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.name == other.name
            && self.vars == other.vars
            && self.clauses == other.clauses
            && self.fail == other.fail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IIf {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub guards: Vec<IExpr>,
    pub then_body: Vec<IExpr>,
    pub else_body: Vec<IExpr>,
}
annotated!(IIf);
impl Eq for IIf {}
impl PartialEq for IIf {
    fn eq(&self, other: &Self) -> bool {
        self.guards == other.guards
            && self.then_body == other.then_body
            && self.else_body == other.else_body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ILetRec {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub defs: Vec<(Var, IExpr)>,
    pub body: Vec<IExpr>,
}
annotated!(ILetRec);
impl Eq for ILetRec {}
impl PartialEq for ILetRec {
    fn eq(&self, other: &Self) -> bool {
        self.defs == other.defs && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IMatch {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub pattern: Box<IExpr>,
    pub guards: Vec<IExpr>,
    pub arg: Box<IExpr>,
    pub fail: Box<IClause>,
}
annotated!(IMatch);
impl Eq for IMatch {}
impl PartialEq for IMatch {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.guards == other.guards
            && self.arg == other.arg
            && self.fail == other.fail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IMap {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<IExpr>,
    pub pairs: Vec<IMapPair>,
    pub is_pattern: bool,
}
annotated!(IMap);
impl Eq for IMap {}
impl PartialEq for IMap {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.pairs == other.pairs && self.is_pattern == other.is_pattern
    }
}
impl IMap {
    pub fn new(span: SourceSpan, pairs: Vec<IMapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(IExpr::Literal(Literal {
                span,
                annotations: Annotations::default(),
                value: Lit::Map(Default::default()),
            })),
            pairs,
            is_pattern: false,
        }
    }

    pub fn new_pattern(span: SourceSpan, pairs: Vec<IMapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(IExpr::Literal(Literal {
                span,
                annotations: Annotations::default(),
                value: Lit::Map(Default::default()),
            })),
            pairs,
            is_pattern: true,
        }
    }

    pub fn update(span: SourceSpan, map: IExpr, pairs: Vec<IMapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(map),
            pairs,
            is_pattern: false,
        }
    }

    pub fn is_literal(&self) -> bool {
        self.arg.is_literal()
            && self
                .pairs
                .iter()
                .all(|p| p.key.len() == 1 && p.key[0].is_literal() && p.value.is_literal())
    }

    pub fn is_simple(&self) -> bool {
        self.arg.is_simple()
            && self
                .pairs
                .iter()
                .all(|pair| pair.key.iter().all(|k| k.is_simple()) && pair.value.is_simple())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IMapPair {
    pub op: MapOp,
    pub key: Vec<IExpr>,
    pub value: Box<IExpr>,
}

#[derive(Debug, Clone, Spanned)]
pub struct IPrimOp {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: Symbol,
    pub args: Vec<IExpr>,
}
annotated!(IPrimOp);
impl IPrimOp {
    pub fn new(span: SourceSpan, name: Symbol, args: Vec<IExpr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            name,
            args,
        }
    }
}
impl Eq for IPrimOp {}
impl PartialEq for IPrimOp {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IProtect {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub body: Vec<IExpr>,
}
annotated!(IProtect);
impl Eq for IProtect {}
impl PartialEq for IProtect {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IReceive1 {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub clauses: Vec<IClause>,
}
annotated!(IReceive1);
impl Eq for IReceive1 {}
impl PartialEq for IReceive1 {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IReceive2 {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub clauses: Vec<IClause>,
    pub timeout: Box<IExpr>,
    pub action: Vec<IExpr>,
}
annotated!(IReceive2);
impl Eq for IReceive2 {}
impl PartialEq for IReceive2 {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
            && self.timeout == other.timeout
            && self.action == other.action
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ISet {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub var: Var,
    pub arg: Box<IExpr>,
}
annotated!(ISet);
impl ISet {
    pub fn new(span: SourceSpan, var: Var, arg: IExpr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            var,
            arg: Box::new(arg),
        }
    }
}
impl Eq for ISet {}
impl PartialEq for ISet {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.arg == other.arg
    }
}

#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub struct ISimple {
    pub annotations: Annotations,
    #[span]
    pub expr: Box<IExpr>,
}
impl ISimple {
    pub fn new(expr: IExpr) -> Self {
        Self {
            annotations: Annotations::default(),
            expr: Box::new(expr),
        }
    }
}
annotated!(ISimple);

#[derive(Debug, Clone, Default, Spanned)]
pub struct ITuple {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub elements: Vec<IExpr>,
}
annotated!(ITuple);
impl ITuple {
    pub fn new(span: SourceSpan, elements: Vec<IExpr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            elements,
        }
    }
}
impl Eq for ITuple {}
impl PartialEq for ITuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ITry {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub args: Vec<IExpr>,
    pub vars: Vec<Var>,
    pub body: Vec<IExpr>,
    pub evars: Vec<Var>,
    pub handler: Box<IExpr>,
}
annotated!(ITry);
impl Eq for ITry {}
impl PartialEq for ITry {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args
            && self.vars == other.vars
            && self.body == other.body
            && self.evars == other.evars
            && self.handler == other.handler
    }
}

#[derive(Debug, Clone, Spanned)]
pub enum IQualifier {
    Generator(IGen),
    Filter(IFilter),
}

/// This struct is used to represent the internal state of a generator expression while it is being transformed
#[derive(Debug, Clone, Spanned)]
pub struct IGen {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    // acc_pat is the accumulator pattern, e.g. [Pat|Tail] for Pat <- Expr.
    pub acc_pattern: Option<Box<IExpr>>,
    // acc_guard is the list of guards immediately following the current
    // generator in the qualifier list input.
    pub acc_guards: Vec<IExpr>,
    // skip_pat is the skip pattern, e.g. <<X,_:X,Tail/bitstring>> for <<X,1:X>> <= Expr.
    pub skip_pattern: Option<Box<IExpr>>,
    // tail is the variable used in AccPat and SkipPat bound to the rest of the
    // generator input.
    pub tail: Option<Var>,
    // tail_pat is the tail pattern, respectively [] and <<_/bitstring>> for list
    // and bit string generators.
    pub tail_pattern: Box<IExpr>,
    // pre is the list of expressions to be inserted before the comprehension function
    pub pre: Vec<IExpr>,
    // arg is the expression that the comprehension function should be passed
    pub arg: Box<IExpr>,
}
annotated!(IGen);

/// A filter is one of two types of expressions that act as qualifiers in a commprehension, the other is a generator
#[derive(Debug, Clone, Spanned)]
pub struct IFilter {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub filter: FilterType,
}
annotated!(IFilter);
impl IFilter {
    pub fn new_guard(span: SourceSpan, exprs: Vec<IExpr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            filter: FilterType::Guard(exprs),
        }
    }

    pub fn new_match(span: SourceSpan, pre: Vec<IExpr>, matcher: IExpr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            filter: FilterType::Match(pre, Box::new(matcher)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum FilterType {
    Guard(Vec<IExpr>),
    /// Represents a filter expression which lowers to a case
    ///
    /// The first element is used to guarantee that certain expressions
    /// are evaluated before the filter is applied
    ///
    /// The second element is the argument on which the filter is matched
    /// It must be true to accumulate the current value
    /// If false, the current value is skipped
    /// If neither, a bad_filter error is raised
    Match(Vec<IExpr>, Box<IExpr>),
}
