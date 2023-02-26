use std::fmt;

use firefly_binary::BinaryEntrySpecifier;
use firefly_diagnostics::{SourceSpan, Span, Spanned};
use firefly_intern::{symbols, Symbol};
use firefly_syntax_base::*;

macro_rules! impl_expr {
    ($t:ident) => {
        impl Into<Expr> for $t {
            #[inline]
            fn into(self) -> Expr {
                Expr::$t(self)
            }
        }
    };
}

mod function;
mod internal;
mod module;

pub use self::function::*;
pub use self::internal::*;
pub use self::module::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchType {
    Binary,
    BinaryInt,
    BinarySegment,
    BinaryEnd,
    Cons,
    Tuple,
    Map,
    Atom,
    Float,
    Int,
    Nil,
    Literal,
    Var,
}

// Kernel Syntax Tree

// We must define _some_ ordering for Expr in order to ensure that when
// rearranging map keys via sets/maps, that the ordering is consistent
// across compilations. Since keys are defined as Expr, we need to define
// this ordering for all Exprs, but in general we expect such keys to only
// be vars and literals, where literals are ordered in the typical fashion,
// literals are always ordered before vars, and vars are ordered by name. The
// Expr types are ordered relative to their respective definitions,
// i.e. if some type T is defined in the Expr enumeration before some type U,
// then T < U in the ordering, for comparisons between two Ts, we compare them
// based on the expressions contained in their fields, and failing that, whatever
// makes the most sense on a case by case basis.
#[derive(Clone, Spanned, PartialEq, Eq, PartialOrd, Ord)]
pub enum Expr {
    Binary(Binary),
    BinaryInt(BinarySegment),
    BinarySegment(BinarySegment),
    BinaryEnd(SourceSpan),
    Cons(Cons),
    Tuple(Tuple),
    Map(Map),
    Literal(Literal),
    Var(Var),
    Local(Span<FunctionName>),
    Remote(Remote),
    Alias(IAlias),
    Alt(Alt),
    Bif(Bif),
    Break(Break),
    Call(Call),
    Catch(Catch),
    Enter(Enter),
    Fun(IFun),
    Goto(Goto),
    Guard(Guard),
    If(If),
    LetRec(ILetRec),
    LetRecGoto(LetRecGoto),
    Match(Match),
    Put(Put),
    Return(Return),
    Select(Select),
    Seq(Seq),
    Set(ISet),
    Test(Test),
    Try(Try),
    TryEnter(TryEnter),
    Values(IValues),
}
impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Alias(expr) => write!(f, "{:#?}", expr),
            Self::Binary(expr) => write!(f, "{:#?}", expr),
            Self::BinarySegment(expr) => write!(f, "{:#?}", expr),
            Self::BinaryInt(expr) => write!(f, "{:#?}", expr),
            Self::BinaryEnd(_) => write!(f, "BinaryEnd"),
            Self::Cons(expr) => write!(f, "{:#?}", expr),
            Self::Fun(expr) => write!(f, "{:#?}", expr),
            Self::Tuple(expr) => write!(f, "{:#?}", expr),
            Self::Map(expr) => write!(f, "{:#?}", expr),
            Self::Literal(expr) => write!(f, "{:#?}", expr),
            Self::Var(expr) => write!(f, "{:#?}", expr),
            Self::If(expr) => write!(f, "{:#?}", expr),
            Self::Seq(expr) => write!(f, "{:#?}", expr),
            Self::Set(expr) => write!(f, "{:#?}", expr),
            Self::Put(expr) => write!(f, "{:#?}", expr),
            Self::Bif(expr) => write!(f, "{:#?}", expr),
            Self::Test(expr) => write!(f, "{:#?}", expr),
            Self::Guard(expr) => write!(f, "{:#?}", expr),
            Self::Call(expr) => write!(f, "{:#?}", expr),
            Self::Enter(expr) => write!(f, "{:#?}", expr),
            Self::Try(expr) => write!(f, "{:#?}", expr),
            Self::TryEnter(expr) => write!(f, "{:#?}", expr),
            Self::Catch(expr) => write!(f, "{:#?}", expr),
            Self::LetRec(expr) => write!(f, "{:#?}", expr),
            Self::LetRecGoto(expr) => write!(f, "{:#?}", expr),
            Self::Goto(expr) => write!(f, "{:#?}", expr),
            Self::Match(expr) => write!(f, "{:#?}", expr),
            Self::Alt(expr) => write!(f, "{:#?}", expr),
            Self::Select(expr) => write!(f, "{:#?}", expr),
            Self::Break(expr) => write!(f, "{:#?}", expr),
            Self::Return(expr) => write!(f, "{:#?}", expr),
            Self::Values(expr) => write!(f, "{:#?}", expr),
            Self::Local(name) => write!(f, "Local({})", &name.item),
            Self::Remote(expr) => write!(f, "{:#?}", expr),
        }
    }
}
impl Annotated for Expr {
    fn annotations(&self) -> &Annotations {
        match self {
            Self::Alias(expr) => expr.annotations(),
            Self::Binary(expr) => expr.annotations(),
            Self::BinarySegment(expr) => expr.annotations(),
            Self::BinaryInt(expr) => expr.annotations(),
            Self::BinaryEnd(_) => unimplemented!(),
            Self::Cons(expr) => expr.annotations(),
            Self::Fun(expr) => expr.annotations(),
            Self::Tuple(expr) => expr.annotations(),
            Self::Map(expr) => expr.annotations(),
            Self::Literal(expr) => expr.annotations(),
            Self::Var(expr) => expr.annotations(),
            Self::If(expr) => expr.annotations(),
            Self::Seq(expr) => expr.annotations(),
            Self::Set(expr) => expr.annotations(),
            Self::Put(expr) => expr.annotations(),
            Self::Bif(expr) => expr.annotations(),
            Self::Test(expr) => expr.annotations(),
            Self::Guard(expr) => expr.annotations(),
            Self::Call(expr) => expr.annotations(),
            Self::Enter(expr) => expr.annotations(),
            Self::Try(expr) => expr.annotations(),
            Self::TryEnter(expr) => expr.annotations(),
            Self::Catch(expr) => expr.annotations(),
            Self::LetRec(expr) => expr.annotations(),
            Self::LetRecGoto(expr) => expr.annotations(),
            Self::Goto(expr) => expr.annotations(),
            Self::Match(expr) => expr.annotations(),
            Self::Alt(expr) => expr.annotations(),
            Self::Select(expr) => expr.annotations(),
            Self::Break(expr) => expr.annotations(),
            Self::Return(expr) => expr.annotations(),
            Self::Values(expr) => expr.annotations(),
            Self::Local(_) | Self::Remote(_) => unimplemented!(),
        }
    }

    fn annotations_mut(&mut self) -> &mut Annotations {
        match self {
            Self::Alias(expr) => expr.annotations_mut(),
            Self::Binary(expr) => expr.annotations_mut(),
            Self::BinarySegment(expr) => expr.annotations_mut(),
            Self::BinaryInt(expr) => expr.annotations_mut(),
            Self::BinaryEnd(_) => unimplemented!(),
            Self::Cons(expr) => expr.annotations_mut(),
            Self::Fun(expr) => expr.annotations_mut(),
            Self::Tuple(expr) => expr.annotations_mut(),
            Self::Map(expr) => expr.annotations_mut(),
            Self::Literal(expr) => expr.annotations_mut(),
            Self::Var(expr) => expr.annotations_mut(),
            Self::If(expr) => expr.annotations_mut(),
            Self::Seq(expr) => expr.annotations_mut(),
            Self::Set(expr) => expr.annotations_mut(),
            Self::Put(expr) => expr.annotations_mut(),
            Self::Bif(expr) => expr.annotations_mut(),
            Self::Test(expr) => expr.annotations_mut(),
            Self::Guard(expr) => expr.annotations_mut(),
            Self::Call(expr) => expr.annotations_mut(),
            Self::Enter(expr) => expr.annotations_mut(),
            Self::Try(expr) => expr.annotations_mut(),
            Self::TryEnter(expr) => expr.annotations_mut(),
            Self::Catch(expr) => expr.annotations_mut(),
            Self::LetRec(expr) => expr.annotations_mut(),
            Self::LetRecGoto(expr) => expr.annotations_mut(),
            Self::Goto(expr) => expr.annotations_mut(),
            Self::Match(expr) => expr.annotations_mut(),
            Self::Alt(expr) => expr.annotations_mut(),
            Self::Select(expr) => expr.annotations_mut(),
            Self::Break(expr) => expr.annotations_mut(),
            Self::Return(expr) => expr.annotations_mut(),
            Self::Values(expr) => expr.annotations_mut(),
            Self::Local(_) | Self::Remote(_) => unimplemented!(),
        }
    }
}
impl Expr {
    pub fn is_atomic(&self) -> bool {
        match self {
            Self::Literal(_) | Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_literal(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            Self::Cons(cons) => cons.is_literal(),
            Self::Tuple(tuple) => tuple.is_literal(),
            Self::Map(map) => map.is_literal(),
            _ => false,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Self::Literal(lit) => lit.is_integer(),
            _ => false,
        }
    }

    pub fn is_var(&self) -> bool {
        match self {
            Self::Var(_) => true,
            _ => false,
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            Self::Local(_) => true,
            _ => false,
        }
    }

    pub fn as_local(&self) -> Option<FunctionName> {
        match self {
            Self::Local(name) => Some(name.item),
            _ => None,
        }
    }

    pub fn is_bin_end(&self) -> bool {
        match self {
            Self::BinaryEnd(_) => true,
            _ => false,
        }
    }

    ///  Test whether Kexpr is "enterable", i.e. can handle return from
    ///  within itself without extra #k_return{}.
    pub fn is_enter_expr(&self) -> bool {
        match self {
            Self::Try(_) | Self::Call(_) | Self::Match(_) | Self::LetRecGoto(_) => true,
            _ => false,
        }
    }

    pub fn match_type(&self) -> MatchType {
        match self.arg() {
            Self::Cons(_) => MatchType::Cons,
            Self::Tuple(_) => MatchType::Tuple,
            Self::Map(_) => MatchType::Map,
            Self::Binary(_) => MatchType::Binary,
            Self::BinaryInt(_) => MatchType::BinaryInt,
            Self::BinarySegment(_) => MatchType::BinarySegment,
            Self::BinaryEnd(_) => MatchType::BinaryEnd,
            Self::Var(_) => MatchType::Var,
            Self::Literal(Literal {
                value: Lit::Nil, ..
            }) => MatchType::Nil,
            Self::Literal(Literal {
                value: Lit::Atom(_),
                ..
            }) => MatchType::Atom,
            Self::Literal(Literal {
                value: Lit::Integer(_),
                ..
            }) => MatchType::Int,
            Self::Literal(Literal {
                value: Lit::Float(_),
                ..
            }) => MatchType::Float,
            Self::Literal(Literal {
                value: Lit::Cons(_, _),
                ..
            }) => MatchType::Cons,
            Self::Literal(Literal {
                value: Lit::Tuple(_),
                ..
            }) => MatchType::Tuple,
            Self::Literal(_) => MatchType::Literal,
            other => panic!("invalid pattern expression: {:?}", other),
        }
    }

    pub fn arg(&self) -> &Self {
        match self {
            Self::Alias(IAlias { pattern, .. }) => pattern.as_ref(),
            _ => self,
        }
    }

    pub fn into_arg(self) -> Self {
        match self {
            Self::Alias(IAlias { box pattern, .. }) => pattern,
            expr => expr,
        }
    }

    pub fn alias(&self) -> &[Var] {
        match self {
            Self::Alias(IAlias { ref vars, .. }) => vars.as_slice(),
            _ => &[],
        }
    }

    pub fn as_var(&self) -> Option<&Var> {
        match self {
            Self::Var(ref v) => Some(v),
            _ => None,
        }
    }

    pub fn into_var(self) -> Option<Var> {
        match self {
            Self::Var(v) => Some(v),
            _ => None,
        }
    }

    pub fn into_cons(self) -> Option<Cons> {
        match self {
            Self::Cons(cons) => Some(cons),
            _ => None,
        }
    }

    pub fn into_map(self) -> Option<Map> {
        match self {
            Self::Map(map) => Some(map),
            _ => None,
        }
    }

    pub fn into_literal(self) -> Option<Literal> {
        match self {
            Self::Literal(lit) => Some(lit),
            _ => None,
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

    pub fn as_binary(&self) -> Option<&Binary> {
        match self {
            Self::Binary(ref b) => Some(b),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Spanned, PartialEq, Eq, PartialOrd, Ord)]
pub enum Remote {
    Static(Span<FunctionName>),
    Dynamic(#[span] Box<Expr>, Box<Expr>),
}
impl_expr!(Remote);
impl Remote {
    pub fn is_static(&self) -> bool {
        match self {
            Self::Static(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_dynamic(&self) -> bool {
        !self.is_static()
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Binary {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub segment: Box<Expr>,
}
annotated!(Binary);
impl_expr!(Binary);
impl Binary {
    pub fn empty(span: SourceSpan) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            segment: Box::new(Expr::BinaryEnd(span)),
        }
    }

    pub fn new(span: SourceSpan, segment: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            segment: Box::new(segment),
        }
    }
}
impl Eq for Binary {}
impl PartialEq for Binary {
    fn eq(&self, other: &Self) -> bool {
        self.segment == other.segment
    }
}
impl PartialOrd for Binary {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Binary {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.segment.cmp(&other.segment)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct BinarySegment {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub spec: BinaryEntrySpecifier,
    pub size: Option<Box<Expr>>,
    pub value: Box<Expr>,
    pub next: Box<Expr>,
}
annotated!(BinarySegment);
impl BinarySegment {
    /// Returns true if this is the last segment and its size
    /// covers the remaining input, i.e. is None, or the atom 'all'
    pub fn is_all(&self) -> bool {
        if self.next.is_bin_end() {
            return false;
        }
        if self.size.is_none() {
            return true;
        }
        match self.size.as_deref().unwrap() {
            Expr::Literal(Literal {
                value: Lit::Atom(symbols::All),
                ..
            }) => true,
            _ => false,
        }
    }
}
impl Eq for BinarySegment {}
impl PartialEq for BinarySegment {
    fn eq(&self, other: &Self) -> bool {
        self.spec == other.spec
            && self.size == other.size
            && self.value == other.value
            && self.next == other.next
    }
}
impl PartialOrd for BinarySegment {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BinarySegment {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.spec
            .cmp(&other.spec)
            .then_with(|| self.size.cmp(&other.size))
            .then_with(|| self.value.cmp(&other.value))
            .then_with(|| self.next.cmp(&other.next))
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
impl_expr!(Cons);
impl Cons {
    pub fn new(span: SourceSpan, head: Expr, tail: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            head: Box::new(head),
            tail: Box::new(tail),
        }
    }

    pub fn is_literal(&self) -> bool {
        self.head.is_literal() && self.tail.is_literal()
    }
}
impl Eq for Cons {}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head == other.head && self.tail == other.tail
    }
}
impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cons {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.head
            .cmp(&other.head)
            .then_with(|| self.tail.cmp(&other.tail))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Tuple {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub elements: Vec<Expr>,
}
annotated!(Tuple);
impl_expr!(Tuple);
impl Tuple {
    pub fn new(span: SourceSpan, elements: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            elements,
        }
    }

    pub fn is_literal(&self) -> bool {
        self.elements.iter().all(|e| e.is_literal())
    }
}
impl Eq for Tuple {}
impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.elements == other.elements
    }
}
impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tuple {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.elements.cmp(&other.elements)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Map {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub var: Box<Expr>,
    pub op: MapOp,
    pub pairs: Vec<MapPair>,
}
annotated!(Map);
impl_expr!(Map);
impl Map {
    pub fn new(span: SourceSpan, var: Expr, op: MapOp, pairs: Vec<MapPair>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            var: Box::new(var),
            op,
            pairs,
        }
    }

    pub fn is_literal(&self) -> bool {
        self.var.is_literal()
            && self
                .pairs
                .iter()
                .all(|p| p.key.is_literal() && p.value.is_literal())
    }
}
impl Eq for Map {}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.op == other.op && self.pairs == other.pairs
    }
}
impl PartialOrd for Map {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Map {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.op
            .cmp(&other.op)
            .then_with(|| self.var.cmp(&other.var))
            .then_with(|| self.pairs.cmp(&other.pairs))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MapPair {
    pub key: Box<Expr>,
    pub value: Box<Expr>,
}
impl MapPair {
    #[inline]
    pub fn is_var_key(&self) -> bool {
        self.key.is_var()
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct If {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub cond: Box<Expr>,
    pub then_body: Box<Expr>,
    pub else_body: Box<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(If);
impl_expr!(If);
impl If {
    pub fn new(span: SourceSpan, cond: Expr, then_body: Expr, else_body: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            cond: Box::new(cond),
            then_body: Box::new(then_body),
            else_body: Box::new(else_body),
            ret: vec![],
        }
    }
}
impl Eq for If {}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.cond == other.cond
            && self.then_body == other.then_body
            && self.else_body == other.else_body
            && self.ret == other.ret
    }
}
impl PartialOrd for If {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for If {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cond
            .cmp(&other.cond)
            .then_with(|| self.then_body.cmp(&other.then_body))
            .then_with(|| self.else_body.cmp(&other.else_body))
            .then_with(|| self.ret.cmp(&other.ret))
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
impl_expr!(Seq);
impl Seq {
    pub fn new(span: SourceSpan, arg: Expr, body: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(arg),
            body: Box::new(body),
        }
    }
}
impl Eq for Seq {}
impl PartialEq for Seq {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.body == other.body
    }
}
impl PartialOrd for Seq {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Seq {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.arg
            .cmp(&other.arg)
            .then_with(|| self.body.cmp(&other.body))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Put {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(Put);
impl_expr!(Put);
impl Put {
    pub fn new(span: SourceSpan, arg: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            arg: Box::new(arg),
            ret: vec![],
        }
    }
}
impl Eq for Put {}
impl PartialEq for Put {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg && self.ret == other.ret
    }
}
impl PartialOrd for Put {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Put {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.arg
            .cmp(&other.arg)
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Bif {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub op: FunctionName,
    pub args: Vec<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(Bif);
impl_expr!(Bif);
impl Bif {
    pub fn new(span: SourceSpan, op: FunctionName, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            op,
            args,
            ret: vec![],
        }
    }

    pub fn is_type_test(&self) -> bool {
        self.op.is_type_test()
    }

    pub fn is_comp_op(&self) -> bool {
        self.op.is_comparison_op()
    }
}
impl Eq for Bif {}
impl PartialEq for Bif {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.args == other.args && self.ret == other.ret
    }
}
impl PartialOrd for Bif {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bif {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.op
            .cmp(&other.op)
            .then_with(|| self.args.cmp(&other.args))
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Test {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub op: FunctionName,
    pub args: Vec<Expr>,
}
annotated!(Test);
impl_expr!(Test);
impl Test {
    pub fn new(span: SourceSpan, op: FunctionName, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            op,
            args,
        }
    }
}
impl Eq for Test {}
impl PartialEq for Test {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.args == other.args
    }
}
impl PartialOrd for Test {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Test {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.op
            .cmp(&other.op)
            .then_with(|| self.args.cmp(&other.args))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Call {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    // Expected to be either Expr::Remote or Expr::Local after lowering
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(Call);
impl_expr!(Call);
impl Call {
    pub fn new(span: SourceSpan, callee: FunctionName, args: Vec<Expr>) -> Self {
        let callee = if callee.is_local() {
            Box::new(Expr::Local(Span::new(span, callee)))
        } else {
            Box::new(Expr::Remote(Remote::Static(Span::new(span, callee))))
        };
        Self {
            span,
            annotations: Annotations::default(),
            callee,
            args,
            ret: vec![],
        }
    }

    pub fn static_callee(&self) -> Option<FunctionName> {
        match self.callee.as_ref() {
            Expr::Local(local) => Some(local.item),
            Expr::Remote(Remote::Static(remote)) => Some(remote.item),
            _ => None,
        }
    }

    pub fn module_symbol(&self) -> Option<Symbol> {
        match self.callee.as_ref() {
            Expr::Remote(Remote::Static(remote)) => remote.module,
            Expr::Remote(Remote::Dynamic(
                box Expr::Literal(Literal {
                    value: Lit::Atom(m),
                    ..
                }),
                _,
            )) => Some(*m),
            _ => None,
        }
    }

    pub fn function_symbol(&self) -> Option<Symbol> {
        match self.callee.as_ref() {
            Expr::Local(local) => Some(local.function),
            Expr::Remote(Remote::Static(remote)) => Some(remote.function),
            Expr::Remote(Remote::Dynamic(
                _,
                box Expr::Literal(Literal {
                    value: Lit::Atom(f),
                    ..
                }),
            )) => Some(*f),
            _ => None,
        }
    }
}
impl Eq for Call {}
impl PartialEq for Call {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args && self.ret == other.ret
    }
}
impl PartialOrd for Call {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Call {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.callee
            .cmp(&other.callee)
            .then_with(|| self.args.cmp(&other.args))
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Enter {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub callee: Box<Expr>,
    pub args: Vec<Expr>,
}
annotated!(Enter);
impl_expr!(Enter);
impl Enter {
    pub fn new(span: SourceSpan, callee: Expr, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            callee: Box::new(callee),
            args,
        }
    }

    pub fn static_callee(&self) -> Option<FunctionName> {
        match self.callee.as_ref() {
            Expr::Local(local) => Some(local.item),
            Expr::Remote(Remote::Static(remote)) => Some(remote.item),
            _ => None,
        }
    }

    pub fn module_symbol(&self) -> Option<Symbol> {
        match self.callee.as_ref() {
            Expr::Remote(Remote::Static(remote)) => remote.module,
            Expr::Remote(Remote::Dynamic(
                box Expr::Literal(Literal {
                    value: Lit::Atom(m),
                    ..
                }),
                _,
            )) => Some(*m),
            _ => None,
        }
    }

    pub fn function_symbol(&self) -> Option<Symbol> {
        match self.callee.as_ref() {
            Expr::Local(local) => Some(local.function),
            Expr::Remote(Remote::Static(remote)) => Some(remote.function),
            Expr::Remote(Remote::Dynamic(
                _,
                box Expr::Literal(Literal {
                    value: Lit::Atom(f),
                    ..
                }),
            )) => Some(*f),
            _ => None,
        }
    }
}
impl Eq for Enter {}
impl PartialEq for Enter {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee && self.args == other.args
    }
}
impl PartialOrd for Enter {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Enter {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.callee
            .cmp(&other.callee)
            .then_with(|| self.args.cmp(&other.args))
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
    pub ret: Vec<Expr>,
}
annotated!(Try);
impl_expr!(Try);
impl Eq for Try {}
impl PartialEq for Try {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg
            && self.vars == other.vars
            && self.body == other.body
            && self.evars == other.evars
            && self.handler == other.handler
            && self.ret == other.ret
    }
}
impl PartialOrd for Try {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Try {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.arg
            .cmp(&other.arg)
            .then_with(|| self.vars.cmp(&other.vars))
            .then_with(|| self.body.cmp(&other.body))
            .then_with(|| self.evars.cmp(&other.evars))
            .then_with(|| self.handler.cmp(&other.handler))
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct TryEnter {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub arg: Box<Expr>,
    pub vars: Vec<Var>,
    pub body: Box<Expr>,
    pub evars: Vec<Var>,
    pub handler: Box<Expr>,
}
annotated!(TryEnter);
impl_expr!(TryEnter);
impl Eq for TryEnter {}
impl PartialEq for TryEnter {
    fn eq(&self, other: &Self) -> bool {
        self.arg == other.arg
            && self.vars == other.vars
            && self.body == other.body
            && self.evars == other.evars
            && self.handler == other.handler
    }
}
impl PartialOrd for TryEnter {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TryEnter {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.arg
            .cmp(&other.arg)
            .then_with(|| self.vars.cmp(&other.vars))
            .then_with(|| self.body.cmp(&other.body))
            .then_with(|| self.evars.cmp(&other.evars))
            .then_with(|| self.handler.cmp(&other.handler))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Catch {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub body: Box<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(Catch);
impl_expr!(Catch);
impl Eq for Catch {}
impl PartialEq for Catch {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.ret == other.ret
    }
}
impl PartialOrd for Catch {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Catch {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.body
            .cmp(&other.body)
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct LetRecGoto {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub label: Symbol,
    pub vars: Vec<Var>,
    pub first: Box<Expr>,
    pub then: Box<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(LetRecGoto);
impl_expr!(LetRecGoto);
impl Eq for LetRecGoto {}
impl PartialEq for LetRecGoto {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label
            && self.vars == other.vars
            && self.first == other.first
            && self.then == other.then
            && self.ret == other.ret
    }
}
impl PartialOrd for LetRecGoto {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LetRecGoto {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.label
            .cmp(&other.label)
            .then_with(|| self.vars.cmp(&other.vars))
            .then_with(|| self.first.cmp(&other.first))
            .then_with(|| self.then.cmp(&other.then))
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Goto {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub label: Symbol,
    pub args: Vec<Expr>,
}
annotated!(Goto);
impl_expr!(Goto);
impl Goto {
    pub fn new(span: SourceSpan, label: Symbol, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            label,
            args,
        }
    }
}
impl Eq for Goto {}
impl PartialEq for Goto {
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.args == other.args
    }
}
impl PartialOrd for Goto {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Goto {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.label
            .cmp(&other.label)
            .then_with(|| self.args.cmp(&other.args))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Match {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub body: Box<Expr>,
    pub ret: Vec<Expr>,
}
annotated!(Match);
impl_expr!(Match);
impl Eq for Match {}
impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.ret == other.ret
    }
}
impl PartialOrd for Match {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Match {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.body
            .cmp(&other.body)
            .then_with(|| self.ret.cmp(&other.ret))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Alt {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub first: Box<Expr>,
    pub then: Box<Expr>,
}
annotated!(Alt);
impl_expr!(Alt);
impl Eq for Alt {}
impl PartialEq for Alt {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first && self.then == other.then
    }
}
impl PartialOrd for Alt {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Alt {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.first
            .cmp(&other.first)
            .then_with(|| self.then.cmp(&other.then))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Select {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub var: Var,
    pub types: Vec<TypeClause>,
}
annotated!(Select);
impl_expr!(Select);
impl Eq for Select {}
impl PartialEq for Select {
    fn eq(&self, other: &Self) -> bool {
        self.var == other.var && self.types == other.types
    }
}
impl PartialOrd for Select {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Select {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.var
            .cmp(&other.var)
            .then_with(|| self.types.cmp(&other.types))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct TypeClause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub ty: MatchType,
    pub values: Vec<ValueClause>,
}
annotated!(TypeClause);
impl Eq for TypeClause {}
impl PartialEq for TypeClause {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty && self.values == other.values
    }
}
impl PartialOrd for TypeClause {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TypeClause {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.ty
            .cmp(&other.ty)
            .then_with(|| self.values.cmp(&other.values))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ValueClause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub value: Box<Expr>,
    pub body: Box<Expr>,
}
annotated!(ValueClause);
impl Eq for ValueClause {}
impl PartialEq for ValueClause {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.body == other.body
    }
}
impl PartialOrd for ValueClause {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ValueClause {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value
            .cmp(&other.value)
            .then_with(|| self.body.cmp(&other.body))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Guard {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub clauses: Vec<GuardClause>,
}
annotated!(Guard);
impl Eq for Guard {}
impl PartialEq for Guard {
    fn eq(&self, other: &Self) -> bool {
        self.clauses == other.clauses
    }
}
impl PartialOrd for Guard {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Guard {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.clauses.cmp(&other.clauses)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct GuardClause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub guard: Box<Expr>,
    pub body: Box<Expr>,
}
annotated!(GuardClause);
impl Eq for GuardClause {}
impl PartialEq for GuardClause {
    fn eq(&self, other: &Self) -> bool {
        self.guard == other.guard && self.body == other.body
    }
}
impl PartialOrd for GuardClause {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GuardClause {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.guard
            .cmp(&other.guard)
            .then_with(|| self.body.cmp(&other.body))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Break {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub args: Vec<Expr>,
}
annotated!(Break);
impl_expr!(Break);
impl Break {
    pub fn new(span: SourceSpan, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            args,
        }
    }
}
impl Eq for Break {}
impl PartialEq for Break {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args
    }
}
impl PartialOrd for Break {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Break {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.args.cmp(&other.args)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Return {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub args: Vec<Expr>,
}
annotated!(Return);
impl_expr!(Return);
impl Return {
    pub fn empty(span: SourceSpan) -> Self {
        Self::new(span, vec![])
    }

    pub fn new(span: SourceSpan, args: Vec<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            args,
        }
    }
}
impl Eq for Return {}
impl PartialEq for Return {
    fn eq(&self, other: &Self) -> bool {
        self.args == other.args
    }
}
impl PartialOrd for Return {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Return {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.args.cmp(&other.args)
    }
}
