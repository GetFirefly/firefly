use liblumen_syntax_base::*;

mod function;
mod internal;
mod module;

pub use self::function::*;
pub use self::internal::*;
pub use self::module::*;

use crate::printer::PrettyPrinter;

use std::fmt;

use liblumen_binary::BinaryEntrySpecifier;
use liblumen_diagnostics::{SourceSpan, Span, Spanned};
use liblumen_intern::{symbols, Symbol};

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
