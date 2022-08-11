pub mod printer;

pub use self::printer::PrettyPrinter;

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use rpds::{RedBlackTreeMap, RedBlackTreeSet};

use liblumen_binary::{BinaryEntrySpecifier, BitVec};
use liblumen_diagnostics::{SourceSpan, Span, Spanned};
use liblumen_intern::{symbols, Ident, Symbol};
use liblumen_number::{Float, Integer};
use liblumen_syntax_ssa as syntax_ssa;
use liblumen_util::emit::Emit;

use crate::ast;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Annotation {
    // The common case is just a symbol with significant meaning
    Unit,
    // In some cases we need more flexibility, so we use literals for that purpose,
    // with a key used to unique the annotations for an expression
    Term(Literal),
    // Used for tracking used/new variables associated with expressions
    Vars(RedBlackTreeSet<Ident>),
}
impl From<Literal> for Annotation {
    #[inline]
    fn from(term: Literal) -> Self {
        Self::Term(term)
    }
}
impl From<RedBlackTreeSet<Ident>> for Annotation {
    #[inline]
    fn from(set: RedBlackTreeSet<Ident>) -> Self {
        Self::Vars(set)
    }
}

#[derive(Clone, PartialEq, Eq, Default)]
pub struct Annotations(RedBlackTreeMap<Symbol, Annotation>);
unsafe impl Send for Annotations {}
unsafe impl Sync for Annotations {}
impl fmt::Debug for Annotations {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        f.write_str("@annotations[")?;
        for (key, anno) in self.0.iter() {
            self::printer::print_annotation(f, key, anno)?;
        }
        f.write_char(']')
    }
}
impl Annotations {
    /// Create a new, empty annotation set
    pub fn new() -> Self {
        Self(RedBlackTreeMap::new())
    }

    /// Creates a new annotation set initialized with symbols::CompilerGenerated
    pub fn default_compiler_generated() -> Self {
        let mut this = Self::default();
        this.insert_mut(symbols::CompilerGenerated, Annotation::Unit);
        this
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &Annotation)> {
        self.0.iter()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Clear all annotations
    #[inline]
    pub fn clear(&mut self) {
        self.0 = RedBlackTreeMap::new();
    }

    /// Pushes a new annotation on the set
    #[inline]
    pub fn insert<A: Into<Annotation>>(&self, key: Symbol, anno: A) -> Self {
        Self(self.0.insert(key, anno.into()))
    }

    #[inline]
    pub fn insert_mut<A: Into<Annotation>>(&mut self, key: Symbol, anno: A) {
        self.0.insert_mut(key, anno.into());
    }

    /// Pushes a new unit annotation on the set
    #[inline]
    pub fn set(&mut self, key: Symbol) {
        self.0.insert_mut(key, Annotation::Unit);
    }

    /// Tests if the given annotation key is present in the set
    #[inline]
    pub fn contains(&self, key: Symbol) -> bool {
        self.0.contains_key(&key)
    }

    /// Retrieves an annotation by key
    #[inline]
    pub fn get(&self, key: Symbol) -> Option<&Annotation> {
        self.0.get(&key)
    }

    /// Retrieves an annotation by key, mutably
    #[inline]
    pub fn get_mut(&mut self, key: Symbol) -> Option<&mut Annotation> {
        self.0.get_mut(&key)
    }

    /// Removes an annotation with the given key
    #[inline]
    pub fn remove(&self, key: Symbol) -> Self {
        Self(self.0.remove(&key))
    }

    /// Removes an annotation with the given key
    #[inline]
    pub fn remove_mut(&mut self, key: Symbol) {
        self.0.remove_mut(&key);
    }

    /// Replaces all annotations with the given set
    pub fn replace(&mut self, annotations: Self) {
        self.0 = annotations.0;
    }

    /// Convenience function for accessing the new vars annotation
    pub fn new_vars(&self) -> Option<RedBlackTreeSet<Ident>> {
        match self.get(symbols::New) {
            None => None,
            Some(Annotation::Vars(set)) => Some(set.clone()),
            Some(_) => None,
        }
    }

    /// Convenience function for accessing the used vars annotation
    pub fn used_vars(&self) -> Option<RedBlackTreeSet<Ident>> {
        match self.get(symbols::Used) {
            None => None,
            Some(Annotation::Vars(set)) => Some(set.clone()),
            Some(_) => None,
        }
    }
}
impl From<Symbol> for Annotations {
    fn from(sym: Symbol) -> Self {
        let mut annos = Self::default();
        annos.set(sym);
        annos
    }
}
impl<const N: usize> From<[Symbol; N]> for Annotations {
    #[inline]
    fn from(syms: [Symbol; N]) -> Self {
        Self::from(syms.as_slice())
    }
}
impl<const N: usize> From<[(Symbol, Annotation); N]> for Annotations {
    #[inline]
    fn from(syms: [(Symbol, Annotation); N]) -> Self {
        Self::from(syms.as_slice())
    }
}
impl From<&[Symbol]> for Annotations {
    fn from(syms: &[Symbol]) -> Self {
        let mut annos = Self::default();
        for sym in syms {
            annos.set(*sym);
        }
        annos
    }
}
impl From<&[(Symbol, Annotation)]> for Annotations {
    fn from(syms: &[(Symbol, Annotation)]) -> Self {
        let mut annos = Self::default();
        for (sym, anno) in syms {
            annos.insert_mut(*sym, anno.clone());
        }
        annos
    }
}
impl From<Vec<(Symbol, Annotation)>> for Annotations {
    fn from(mut annos: Vec<(Symbol, Annotation)>) -> Self {
        Self(annos.drain(..).collect())
    }
}

/// This trait is implemented by all types which carry annotations
pub trait Annotated {
    fn annotations(&self) -> &Annotations;
    fn annotations_mut(&mut self) -> &mut Annotations;
    #[inline]
    fn annotate<A: Into<Annotation>>(&mut self, key: Symbol, anno: A) {
        self.annotations_mut().insert_mut(key, anno);
    }
    #[inline]
    fn is_compiler_generated(&self) -> bool {
        self.annotations().contains(symbols::CompilerGenerated)
    }
    #[inline]
    fn mark_compiler_generated(&mut self) {
        self.annotations_mut().set(symbols::CompilerGenerated);
    }
    #[inline]
    fn has_annotation(&self, name: Symbol) -> bool {
        self.annotations().contains(name)
    }
    #[inline]
    fn used_vars(&self) -> RedBlackTreeSet<Ident> {
        self.annotations()
            .used_vars()
            .map(|v| v.clone())
            .unwrap_or_default()
    }
    #[inline]
    fn new_vars(&self) -> RedBlackTreeSet<Ident> {
        self.annotations()
            .new_vars()
            .map(|v| v.clone())
            .unwrap_or_default()
    }
}
impl<T> Annotated for Box<T>
where
    T: Annotated,
{
    fn annotations(&self) -> &Annotations {
        (&**self).annotations()
    }
    fn annotations_mut(&mut self) -> &mut Annotations {
        (&mut **self).annotations_mut()
    }
}

macro_rules! annotated {
    ($t:ident) => {
        impl Annotated for $t {
            fn annotations(&self) -> &Annotations {
                &self.annotations
            }

            fn annotations_mut(&mut self) -> &mut Annotations {
                &mut self.annotations
            }
        }
    };
}

#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub struct Module {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: Ident,
    pub vsn: Option<Literal>,
    pub author: Option<Literal>,
    pub compile: Option<ast::CompileOptions>,
    pub on_load: Option<Span<syntax_ssa::FunctionName>>,
    pub imports: HashMap<syntax_ssa::FunctionName, Span<syntax_ssa::Signature>>,
    pub exports: HashSet<Span<syntax_ssa::FunctionName>>,
    pub behaviours: HashSet<Ident>,
    pub attributes: HashMap<Ident, Expr>,
    pub functions: BTreeMap<syntax_ssa::FunctionName, Function>,
}
annotated!(Module);
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut pp = PrettyPrinter::new(f);
        pp.print_module(self)
    }
}
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("cst")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;

        write!(f, "{}", self)?;
        Ok(())
    }
}

/// These are expression types that are used internally during translation from AST to CST
///
/// The final CST never contains these expression types.
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
        let mut pp = PrettyPrinter::new(f);
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
    pub fn new(bodies: Vec<Vec<IExpr>>) -> Self {
        Self {
            span: SourceSpan::UNKNOWN,
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
        let mut pp = PrettyPrinter::new(f);
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

/// A CST expression
#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub enum Expr {
    Alias(Alias),
    Apply(Apply),
    Binary(Binary),
    Call(Call),
    Case(Case),
    Catch(Catch),
    Cons(Cons),
    Fun(Fun),
    If(If),
    Let(Let),
    LetRec(LetRec),
    Literal(Literal),
    Map(Map),
    PrimOp(PrimOp),
    Receive(Receive),
    Seq(Seq),
    Try(Try),
    Tuple(Tuple),
    Values(Values),
    Var(Var),
}
impl Annotated for Expr {
    fn annotations(&self) -> &Annotations {
        match self {
            Self::Alias(expr) => expr.annotations(),
            Self::Apply(expr) => expr.annotations(),
            Self::Binary(expr) => expr.annotations(),
            Self::Call(expr) => expr.annotations(),
            Self::Case(expr) => expr.annotations(),
            Self::Catch(expr) => expr.annotations(),
            Self::Cons(expr) => expr.annotations(),
            Self::Fun(expr) => expr.annotations(),
            Self::If(expr) => expr.annotations(),
            Self::Let(expr) => expr.annotations(),
            Self::LetRec(expr) => expr.annotations(),
            Self::Literal(expr) => expr.annotations(),
            Self::Map(expr) => expr.annotations(),
            Self::PrimOp(expr) => expr.annotations(),
            Self::Receive(expr) => expr.annotations(),
            Self::Seq(expr) => expr.annotations(),
            Self::Try(expr) => expr.annotations(),
            Self::Tuple(expr) => expr.annotations(),
            Self::Values(expr) => expr.annotations(),
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
            Self::Fun(expr) => expr.annotations_mut(),
            Self::If(expr) => expr.annotations_mut(),
            Self::Let(expr) => expr.annotations_mut(),
            Self::LetRec(expr) => expr.annotations_mut(),
            Self::Literal(expr) => expr.annotations_mut(),
            Self::Map(expr) => expr.annotations_mut(),
            Self::PrimOp(expr) => expr.annotations_mut(),
            Self::Receive(expr) => expr.annotations_mut(),
            Self::Seq(expr) => expr.annotations_mut(),
            Self::Try(expr) => expr.annotations_mut(),
            Self::Tuple(expr) => expr.annotations_mut(),
            Self::Values(expr) => expr.annotations_mut(),
            Self::Var(expr) => expr.annotations_mut(),
        }
    }
}
impl Expr {
    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Literal(_) | Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_simple(&self) -> bool {
        match self {
            Self::Var(_) | Self::Literal(_) => true,
            Self::Cons(Cons { head, tail, .. }) => head.is_simple() && tail.is_simple(),
            Self::Tuple(Tuple { elements, .. }) => elements.iter().all(|e| e.is_simple()),
            Self::Map(Map { pairs, .. }) => pairs
                .iter()
                .all(|pair| pair.key.is_simple() && pair.value.is_simple()),
            _ => false,
        }
    }

    pub fn is_safe(&self) -> bool {
        match self {
            Self::Cons(_) | Self::Tuple(_) | Self::Literal(_) => true,
            Self::Var(v) => v.arity.is_none(),
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

    pub fn is_integer(&self) -> bool {
        match self {
            Self::Literal(Literal {
                value: Lit::Integer(_),
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

    pub fn is_var_used(&self, var: &Var) -> bool {
        match self {
            Self::Alias(Alias { var: var2, .. }) => var2.name() == var.name(),
            Self::Values(Values { values, .. }) => values.iter().any(|v| v.is_var_used(var)),
            Self::Var(v) => v.name() == var.name(),
            Self::Literal(_) => false,
            Self::Cons(Cons { head, tail, .. }) => head.is_var_used(var) || tail.is_var_used(var),
            Self::Tuple(Tuple { elements, .. }) => elements.iter().any(|v| v.is_var_used(var)),
            Self::Map(Map { arg, pairs, .. }) => {
                arg.is_var_used(var)
                    || pairs
                        .iter()
                        .any(|p| p.key.is_var_used(var) || p.value.is_var_used(var))
            }
            Self::Binary(Binary { segments, .. }) => segments.iter().any(|s| {
                s.value.is_var_used(var)
                    || s.size
                        .as_ref()
                        .map(|sz| sz.is_var_used(var))
                        .unwrap_or_default()
            }),
            Self::Fun(Fun { vars, body, .. }) => {
                // Variables in fun shadow previous variables
                if vars.contains(var) {
                    false
                } else {
                    body.is_var_used(var)
                }
            }
            Self::If(If {
                guard,
                then_body,
                else_body,
                ..
            }) => {
                guard.is_var_used(var) || then_body.is_var_used(var) || else_body.is_var_used(var)
            }
            Self::Let(Let {
                vars, arg, body, ..
            }) => {
                if arg.is_var_used(var) {
                    true
                } else {
                    // Variables in let shadow previous variables
                    if vars.contains(var) {
                        false
                    } else {
                        body.is_var_used(var)
                    }
                }
            }
            Self::LetRec(LetRec { defs, body, .. }) => {
                defs.iter().any(|(_, v)| v.is_var_used(var)) || body.is_var_used(var)
            }
            Self::Seq(Seq { arg, body, .. }) => arg.is_var_used(var) || body.is_var_used(var),
            Self::Case(Case { arg, clauses, .. }) => {
                if arg.is_var_used(var) {
                    true
                } else {
                    clauses.iter().any(|clause| {
                        clause
                            .patterns
                            .iter()
                            .any(|p| p.is_var_used_in_pattern(var))
                            || clause
                                .guard
                                .as_ref()
                                .map(|g| g.is_var_used(var))
                                .unwrap_or_default()
                            || clause.body.is_var_used(var)
                    })
                }
            }
            Self::Apply(Apply { callee, args, .. }) => {
                callee.is_var_used(var) || args.iter().any(|a| a.is_var_used(var))
            }
            Self::Call(Call {
                module,
                function,
                args,
                ..
            }) => {
                module.is_var_used(var)
                    || function.is_var_used(var)
                    || args.iter().any(|a| a.is_var_used(var))
            }
            Self::PrimOp(PrimOp { args, .. }) => args.iter().any(|v| v.is_var_used(var)),
            Self::Catch(Catch { body, .. }) => body.is_var_used(var),
            Self::Try(Try {
                arg,
                vars,
                evars,
                handler,
                ..
            }) => {
                if arg.is_var_used(var) {
                    true
                } else {
                    // Variables shadow previous ones
                    let is_shadowed = vars.contains(var) || evars.contains(var);
                    if is_shadowed {
                        false
                    } else {
                        handler.is_var_used(var)
                    }
                }
            }
            other => unimplemented!("variable analysis for {:?}", other),
        }
    }

    fn is_var_used_in_pattern(&self, var: &Var) -> bool {
        match self {
            Self::Var(v) => v.name() == var.name(),
            Self::Cons(Cons { head, tail, .. }) => {
                head.is_var_used_in_pattern(var) || tail.is_var_used_in_pattern(var)
            }
            Self::Tuple(Tuple { elements, .. }) => {
                elements.iter().any(|e| e.is_var_used_in_pattern(var))
            }
            Self::Binary(Binary { segments, .. }) => segments.iter().any(|s| {
                s.size
                    .as_ref()
                    .map(|sz| sz.is_var_used_in_pattern(var))
                    .unwrap_or_default()
            }),
            Self::Map(Map { pairs, .. }) => pairs
                .iter()
                .any(|p| p.key.is_var_used(var) || p.value.is_var_used_in_pattern(var)),
            Self::Literal(_) => false,
            other => unimplemented!("pattern variable analysis for {:?}", other),
        }
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Alias {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub var: Var,
    pub pattern: Box<Expr>,
}
annotated!(Alias);
impl Alias {
    pub fn new(span: SourceSpan, var: Var, pattern: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            var,
            pattern: Box::new(pattern),
        }
    }
}
impl Eq for Alias {}
impl PartialEq for Alias {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.pattern == other.pattern
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Apply {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}
annotated!(Apply);
impl Apply {
    pub fn new(span: SourceSpan, callee: Expr, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            callee: Box::new(callee),
            args,
        }
    }
}
impl Eq for Apply {}
impl PartialEq for Apply {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Binary {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub segments: Vec<Bitstring>,
}
annotated!(Binary);
impl Binary {
    pub fn new(span: SourceSpan, segments: Vec<Bitstring>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            segments,
        }
    }
}
impl Eq for Binary {}
impl PartialEq for Binary {
    fn eq(&self, other: &Self) -> bool {
        self.segments == other.segments
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Bitstring {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub value: Box<Expr>,
    pub size: Option<Box<Expr>>,
    pub spec: BinaryEntrySpecifier,
}
annotated!(Bitstring);
impl Eq for Bitstring {}
impl PartialEq for Bitstring {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.size == other.size && self.spec == other.spec
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Call {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub module: Box<Expr>,
    pub function: Box<Expr>,
    pub args: Vec<Expr>,
}
annotated!(Call);
impl Call {
    pub fn new(span: SourceSpan, module: Symbol, function: Symbol, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            module: Box::new(Expr::Literal(Literal::atom(span, module))),
            function: Box::new(Expr::Literal(Literal::atom(span, function))),
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
impl Eq for Call {}
impl PartialEq for Call {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Case {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub clauses: Vec<Clause>,
}
annotated!(Case);
impl Eq for Case {}
impl PartialEq for Case {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Catch {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub body: Box<Expr>,
}
annotated!(Catch);
impl Eq for Catch {}
impl PartialEq for Catch {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Clause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub patterns: Vec<Expr>,
    pub guard: Option<Box<Expr>>,
    pub body: Box<Expr>,
}
annotated!(Clause);
impl Clause {
    pub fn new(span: SourceSpan, patterns: Vec<Expr>, body: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            patterns,
            guard: None,
            body: Box::new(body),
        }
    }
}
impl Eq for Clause {}
impl PartialEq for Clause {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns && self.guard == other.guard && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Cons {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub head: Box<Expr>,
    pub tail: Box<Expr>,
}
annotated!(Cons);
impl Cons {
    pub fn new(span: SourceSpan, head: Expr, tail: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }
}
impl Eq for Cons {}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Function {
    pub var_counter: usize,
    #[span]
    pub fun: Fun,
}
impl Eq for Function {}
impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.fun == other.fun
    }
}
impl Annotated for Function {
    #[inline]
    fn annotations(&self) -> &Annotations {
        self.fun.annotations()
    }

    #[inline]
    fn annotations_mut(&mut self) -> &mut Annotations {
        self.fun.annotations_mut()
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Fun {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: Symbol,
    pub vars: Vec<Var>,
    pub body: Box<Expr>,
}
annotated!(Fun);
impl fmt::Display for Fun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut pp = PrettyPrinter::new(f);
        pp.print_fun(self)
    }
}
impl Eq for Fun {}
impl PartialEq for Fun {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.vars == other.vars && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct If {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub guard: Box<Expr>,
    pub then_body: Box<Expr>,
    pub else_body: Box<Expr>,
}
impl If {
    pub fn new(span: SourceSpan, guard: Expr, t: Expr, f: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            guard: Box::new(guard),
            then_body: Box::new(t),
            else_body: Box::new(f),
        }
    }
}
annotated!(If);
impl Eq for If {}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.guard == other.guard
            && self.then_body == other.then_body
            && self.else_body == other.else_body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Let {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub vars: Vec<Var>,
    pub arg: Box<Expr>,
    pub body: Box<Expr>,
}
annotated!(Let);
impl Let {
    pub fn new(span: SourceSpan, vars: Vec<Var>, arg: Expr, body: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            vars,
            arg: Box::new(arg),
            body: Box::new(body),
        }
    }
}
impl Eq for Let {}
impl PartialEq for Let {
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars && self.arg == other.arg && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct LetRec {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub defs: Vec<(Var, Expr)>,
    pub body: Box<Expr>,
}
annotated!(LetRec);
impl Eq for LetRec {}
impl PartialEq for LetRec {
    fn eq(&self, other: &Self) -> bool {
        self.defs == other.defs && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Literal {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub value: Lit,
}
annotated!(Literal);
impl Literal {
    pub fn atom(span: SourceSpan, sym: Symbol) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Atom(sym),
        }
    }

    pub fn integer<I: Into<Integer>>(span: SourceSpan, i: I) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Integer(i.into()),
        }
    }

    pub fn float<F: Into<Float>>(span: SourceSpan, f: F) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Float(f.into()),
        }
    }

    pub fn nil(span: SourceSpan) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Nil,
        }
    }

    pub fn cons(span: SourceSpan, head: Literal, tail: Literal) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Cons(Box::new(head), Box::new(tail)),
        }
    }

    pub fn tuple(span: SourceSpan, elements: Vec<Literal>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Tuple(elements),
        }
    }

    pub fn map(span: SourceSpan, mut elements: Vec<(Literal, Literal)>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Map(elements.drain(..).collect()),
        }
    }

    pub fn binary(span: SourceSpan, data: BitVec) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            value: Lit::Binary(data),
        }
    }

    pub fn is_integer(&self) -> bool {
        match self.value {
            Lit::Integer(_) => true,
            _ => false,
        }
    }

    pub fn as_integer(&self) -> Option<&Integer> {
        match &self.value {
            Lit::Integer(ref i) => Some(i),
            _ => None,
        }
    }

    pub fn as_atom(&self) -> Option<Symbol> {
        match &self.value {
            Lit::Atom(a) => Some(*a),
            _ => None,
        }
    }
}
impl From<ast::Literal> for Literal {
    fn from(lit: ast::Literal) -> Self {
        match lit {
            ast::Literal::Atom(id) => Self::atom(id.span, id.name),
            ast::Literal::Char(span, c) => Self::integer(span, c as i64),
            ast::Literal::Integer(span, i) => Self::integer(span, i),
            ast::Literal::Float(span, f) => Self::float(span, f),
            ast::Literal::Nil(span) => Self::nil(span),
            ast::Literal::String(id) => {
                let span = id.span;
                id.as_str()
                    .get()
                    .chars()
                    .rev()
                    .map(|c| Self::integer(span, c as i64))
                    .rfold(Self::nil(span), |c, tl| Self::cons(span, c, tl))
            }
            ast::Literal::Cons(span, head, tail) => {
                Self::cons(span, (*head).into(), (*tail).into())
            }
            ast::Literal::Tuple(span, mut elements) => {
                Self::tuple(span, elements.drain(..).map(Literal::from).collect())
            }
            ast::Literal::Map(span, mut map) => {
                let mut new_map: BTreeMap<Literal, Literal> = BTreeMap::new();
                while let Some((k, v)) = map.pop_first() {
                    new_map.insert(k.into(), v.into());
                }
                Self {
                    span,
                    annotations: Annotations::default(),
                    value: Lit::Map(new_map),
                }
            }
            ast::Literal::Binary(span, bin) => Self {
                span,
                annotations: Annotations::default(),
                value: Lit::Binary(bin),
            },
        }
    }
}
impl PartialEq for Literal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for Literal {}
impl Hash for Literal {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}
impl PartialOrd for Literal {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl Ord for Literal {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Lit {
    Atom(Symbol),
    Integer(Integer),
    Float(Float),
    Nil,
    Cons(Box<Literal>, Box<Literal>),
    Tuple(Vec<Literal>),
    Map(BTreeMap<Literal, Literal>),
    Binary(BitVec),
}
impl fmt::Debug for Lit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self::printer::print_lit(f, self)
    }
}
impl Lit {
    pub fn is_number(&self) -> bool {
        match self {
            Self::Integer(_) | Self::Float(_) => true,
            _ => false,
        }
    }
}
impl Hash for Lit {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Atom(x) => x.hash(state),
            Self::Float(f) => f.hash(state),
            Self::Integer(i) => i.hash(state),
            Self::Nil => (),
            Self::Cons(h, t) => {
                h.hash(state);
                t.hash(state);
            }
            Self::Tuple(elements) => Hash::hash_slice(elements.as_slice(), state),
            Self::Map(map) => map.hash(state),
            Self::Binary(bin) => bin.hash(state),
        }
    }
}
impl PartialOrd for Lit {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Lit {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Float(x), Self::Float(y)) => x.cmp(y),
            (Self::Float(x), Self::Integer(y)) => x.partial_cmp(y).unwrap(),
            (Self::Float(_), _) => Ordering::Less,
            (Self::Integer(x), Self::Integer(y)) => x.cmp(y),
            (Self::Integer(x), Self::Float(y)) => x.partial_cmp(y).unwrap(),
            (Self::Integer(_), _) => Ordering::Less,
            (Self::Atom(_), Self::Float(_)) | (Self::Atom(_), Self::Integer(_)) => {
                Ordering::Greater
            }
            (Self::Atom(x), Self::Atom(y)) => x.cmp(y),
            (Self::Atom(_), _) => Ordering::Less,
            (Self::Tuple(_), Self::Float(_))
            | (Self::Tuple(_), Self::Integer(_))
            | (Self::Tuple(_), Self::Atom(_)) => Ordering::Greater,
            (Self::Tuple(xs), Self::Tuple(ys)) => xs.cmp(ys),
            (Self::Tuple(_), _) => Ordering::Less,
            (Self::Map(_), Self::Float(_))
            | (Self::Map(_), Self::Integer(_))
            | (Self::Map(_), Self::Atom(_))
            | (Self::Map(_), Self::Tuple(_)) => Ordering::Greater,
            (Self::Map(x), Self::Map(y)) => x.cmp(y),
            (Self::Map(_), _) => Ordering::Less,
            (Self::Nil, Self::Nil) => Ordering::Equal,
            (Self::Nil, Self::Cons(_, _)) => Ordering::Less,
            (Self::Nil, _) => Ordering::Greater,
            (Self::Cons(h1, t1), Self::Cons(h2, t2)) => match h1.cmp(&h2) {
                Ordering::Equal => t1.cmp(&t2),
                other => other,
            },
            (Self::Cons(_, _), Self::Binary(_)) => Ordering::Less,
            (Self::Cons(_, _), _) => Ordering::Greater,
            (Self::Binary(x), Self::Binary(y)) => x.cmp(y),
            (Self::Binary(_), _) => Ordering::Greater,
        }
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Map {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub pairs: Vec<MapPair>,
    pub is_pattern: bool,
}
annotated!(Map);
impl Eq for Map {}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.pairs == other.pairs && self.is_pattern == other.is_pattern
    }
}
impl Map {
    pub fn new(span: SourceSpan, pairs: Vec<MapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(Expr::Literal(Literal {
                span,
                annotations: Annotations::default(),
                value: Lit::Map(Default::default()),
            })),
            pairs,
            is_pattern: false,
        }
    }

    pub fn new_pattern(span: SourceSpan, pairs: Vec<MapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(Expr::Literal(Literal {
                span,
                annotations: Annotations::default(),
                value: Lit::Map(Default::default()),
            })),
            pairs,
            is_pattern: true,
        }
    }

    pub fn update(span: SourceSpan, map: Expr, pairs: Vec<MapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(map),
            pairs,
            is_pattern: false,
        }
    }

    pub fn pattern(mut self) -> Self {
        self.is_pattern = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapPair {
    pub op: MapOp,
    pub key: Box<Expr>,
    pub value: Box<Expr>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MapOp {
    Assoc,
    Exact,
}

#[derive(Debug, Clone, Spanned)]
pub struct PrimOp {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: Symbol,
    pub args: Vec<Expr>,
}
annotated!(PrimOp);
impl PrimOp {
    pub fn new(span: SourceSpan, name: Symbol, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            name,
            args,
        }
    }
}
impl Eq for PrimOp {}
impl PartialEq for PrimOp {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.args == other.args
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Receive {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub clauses: Vec<Clause>,
    pub timeout: Box<Expr>,
    pub action: Box<Expr>,
}
annotated!(Receive);
impl Eq for Receive {}
impl PartialEq for Receive {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
            && self.timeout == other.timeout
            && self.action == other.action
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Seq {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub body: Box<Expr>,
}
annotated!(Seq);
impl Seq {
    pub fn new(span: SourceSpan, first: Expr, second: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(first),
            body: Box::new(second),
        }
    }
}
impl Eq for Seq {}
impl PartialEq for Seq {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.body == other.body
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Try {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub vars: Vec<Var>,
    pub body: Box<Expr>,
    pub evars: Vec<Var>,
    pub handler: Box<Expr>,
}
annotated!(Try);
impl Eq for Try {}
impl PartialEq for Try {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg
            && self.vars == other.vars
            && self.body == other.body
            && self.evars == other.evars
            && self.handler == other.handler
    }
}

#[derive(Debug, Clone, Default, Spanned)]
pub struct Tuple {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub elements: Vec<Expr>,
}
annotated!(Tuple);
impl Tuple {
    pub fn new(span: SourceSpan, elements: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            elements,
        }
    }
}
impl Eq for Tuple {}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}

#[derive(Debug, Clone, Default, Spanned)]
pub struct Values {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub values: Vec<Expr>,
}
annotated!(Values);
impl Values {
    pub fn new(span: SourceSpan, mut values: Vec<Expr>) -> Expr {
        if values.is_empty() {
            return Expr::Values(Self {
                span,
                annotations: Annotations::default(),
                values,
            });
        }
        if values.len() == 1 {
            return values.pop().unwrap();
        }
        let annotations = values[0].annotations().clone();
        Expr::Values(Self {
            span,
            annotations,
            values,
        })
    }
}
impl Eq for Values {}
impl PartialEq for Values {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

#[derive(Clone, Spanned)]
pub struct Var {
    pub annotations: Annotations,
    #[span]
    pub name: Ident,
    /// Used to represent function variables
    pub arity: Option<usize>,
}
annotated!(Var);
impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;

        match self.arity {
            None => write!(f, "Var({}", self.name.name)?,
            Some(arity) => write!(f, "Var({}/{}", self.name.name, arity)?,
        }
        if self.annotations.is_empty() {
            f.write_char(')')
        } else {
            write!(f, " {:?})", &self.annotations)
        }
    }
}
impl Var {
    pub fn new(name: Ident) -> Self {
        Self {
            annotations: Annotations::default(),
            name,
            arity: None,
        }
    }

    pub fn new_with_arity(name: Ident, arity: usize) -> Self {
        Self {
            annotations: Annotations::default(),
            name,
            arity: Some(arity),
        }
    }

    pub fn name(&self) -> Symbol {
        self.name.name
    }

    #[inline]
    pub fn is_wildcard(&self) -> bool {
        self.name == symbols::Underscore
    }
}
impl Eq for Var {}
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.arity == other.arity
    }
}
impl PartialOrd for Var {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Var {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.name.cmp(&other.name) {
            Ordering::Equal if self.arity.is_none() && other.arity.is_none() => Ordering::Equal,
            Ordering::Equal if self.arity.is_some() && other.arity.is_some() => {
                self.arity.unwrap().cmp(&other.arity.unwrap())
            }
            other => other,
        }
    }
}
