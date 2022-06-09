use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use liblumen_diagnostics::{Diagnostic, Label, Reporter, SourceSpan, Spanned};
use liblumen_syntax_core::{self as syntax_core};

use super::{Arity, Expr, Ident, Name, TypeSpec};

/// A top-level function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub span: SourceSpan,
    pub name: Ident,
    pub arity: u8,
    pub spec: Option<TypeSpec>,
    pub is_nif: bool,
    pub clauses: Vec<FunctionClause>,
}
impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.arity == other.arity
            && self.clauses == other.clauses
            && self.spec == other.spec
            && self.is_nif == other.is_nif
    }
}
impl Function {
    pub fn local_name(&self) -> syntax_core::FunctionName {
        syntax_core::FunctionName::new_local(self.name.name, self.arity)
    }

    pub fn new(
        reporter: &Reporter,
        span: SourceSpan,
        clauses: Vec<FunctionClause>,
    ) -> Result<Self, ()> {
        debug_assert!(clauses.len() > 0);

        let head = &clauses[0];
        let head_span = head.span.clone();
        let name = head.name.unwrap().atom();
        let arity = head.params.len().try_into().unwrap();

        // Check clauses
        let mut last_clause = head_span.clone();
        for clause in clauses.iter().skip(1) {
            if clause.name.is_none() {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("expected named function clause")
                        .with_labels(vec![
                            Label::primary(clause.span.source_id(), clause.span).with_message(
                                "this clause has no name, but a name is required here",
                            ),
                            Label::secondary(last_clause.source_id(), last_clause).with_message(
                                "expected a clause with the same name as this clause",
                            ),
                        ]),
                );
                return Err(());
            }

            let clause_name = clause.name.clone().unwrap();
            let clause_arity: u8 = clause.params.len().try_into().unwrap();

            if clause_name != Name::Atom(name) {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("unterminated function clause")
                        .with_labels(vec![
                            Label::primary(last_clause.source_id(), last_clause.clone())
                                .with_message(
                                "this clause ends with ';', indicating that another clause follows",
                            ),
                            Label::secondary(clause.span.source_id(), clause.span.clone())
                                .with_message("but this clause has a different name"),
                        ]),
                );
                continue;
            }
            if clause_arity != arity {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("unterminated function clause")
                        .with_labels(vec![
                            Label::primary(last_clause.source_id(), last_clause.clone())
                                .with_message(
                                "this clause ends with ';', indicating that another clause follows",
                            ),
                            Label::secondary(clause.span.source_id(), clause.span.clone())
                                .with_message("but this clause has a different arity"),
                        ]),
                );
                continue;
            }

            last_clause = clause.span.clone();
        }

        Ok(Self {
            span,
            name,
            arity,
            clauses,
            spec: None,
            is_nif: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FunctionClause {
    pub span: SourceSpan,
    pub name: Option<Name>,
    pub params: Vec<Expr>,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}
impl PartialEq for FunctionClause {
    fn eq(&self, other: &FunctionClause) -> bool {
        self.name == other.name
            && self.params == other.params
            && self.guard == other.guard
            && self.body == other.body
    }
}
impl FunctionClause {
    pub fn new(
        span: SourceSpan,
        name: Option<Name>,
        params: Vec<Expr>,
        guard: Option<Vec<Guard>>,
        body: Vec<Expr>,
    ) -> Self {
        FunctionClause {
            span,
            name,
            params,
            guard,
            body,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Guard {
    pub span: SourceSpan,
    pub conditions: Vec<Expr>,
}
impl Guard {
    /// Returns `Some(bool)` if the guard consists of a single expression that is a literal boolean
    pub fn as_boolean(&self) -> Option<bool> {
        if self.conditions.len() > 1 {
            return None;
        }
        self.conditions[0].as_boolean()
    }
}
impl PartialEq for Guard {
    fn eq(&self, other: &Guard) -> bool {
        self.conditions == other.conditions
    }
}

/// An anonymous function that can refer to itself within its own body, e.g.:
///
/// ```erlang
/// AnonFib = fun Fib(X) when X < 2 -> 1; Fib(X) -> Fib(X-1) + Fib(X-2).
/// AnonFib(5)
/// ```
#[derive(Debug, Clone)]
pub struct RecursiveFun {
    pub span: SourceSpan,
    // Name is only set when an anonymous function is assigned a name by a compiler pass
    // Immediately after parsing, it is always None
    pub name: Option<Ident>,
    // The self_name is the name bound to the function within its body, which allows the function to call itself
    pub self_name: Ident,
    pub arity: u8,
    pub clauses: Vec<FunctionClause>,
}
impl PartialEq for RecursiveFun {
    fn eq(&self, other: &Self) -> bool {
        self.self_name == other.self_name
            && self.arity == other.arity
            && self.clauses == other.clauses
    }
}

/// An anonymous function that cannot refer to itself within its body
#[derive(Debug, Clone)]
pub struct AnonymousFun {
    pub span: SourceSpan,
    // Name is only set when an anonymous function is assigned a name by a compiler pass
    // Immediately after parsing, it is always None
    pub name: Option<Ident>,
    pub arity: u8,
    pub clauses: Vec<FunctionClause>,
}
impl PartialEq for AnonymousFun {
    fn eq(&self, other: &Self) -> bool {
        self.arity == other.arity && self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Fun {
    Recursive(RecursiveFun),
    Anonymous(AnonymousFun),
}
impl Fun {
    pub fn span(&self) -> SourceSpan {
        match self {
            Self::Recursive(RecursiveFun { ref span, .. }) => span.clone(),
            Self::Anonymous(AnonymousFun { ref span, .. }) => span.clone(),
        }
    }

    pub fn new(
        reporter: &Reporter,
        span: SourceSpan,
        clauses: Vec<FunctionClause>,
    ) -> Result<Self, ()> {
        debug_assert!(clauses.len() > 0);
        let head = &clauses[0];
        let name = head.name.clone();
        let head_span = head.span.clone();
        let arity = head.params.len().try_into().unwrap();

        // Check clauses
        let mut last_clause = head_span.clone();
        for clause in clauses.iter().skip(1) {
            if name.is_some() && clause.name.is_none() {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("expected named function clause")
                        .with_labels(vec![
                            Label::primary(clause.span.source_id(), clause.span.clone())
                                .with_message(
                                    "this clause has no name, but a name is required here",
                                ),
                            Label::secondary(last_clause.source_id(), last_clause).with_message(
                                "expected a clause with the same name as this clause",
                            ),
                        ]),
                );
                return Err(());
            }

            if name.is_none() && clause.name.is_some() {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("mismatched function clause")
                        .with_labels(vec![
                            Label::primary(clause.span.source_id(), clause.span.clone())
                                .with_message("this clause is named"),
                            Label::secondary(last_clause.source_id(), last_clause.clone())
                                .with_message(
                                "but this clause is unnamed, all clauses must share the same name",
                            ),
                        ]),
                );
                return Err(());
            }

            if clause.name != name {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("unterminated function clause")
                        .with_labels(vec![
                            Label::primary(last_clause.source_id(), last_clause.clone())
                                .with_message(
                                "this clause ends with ';', indicating that another clause follows",
                            ),
                            Label::secondary(clause.span.source_id(), clause.span.clone())
                                .with_message("but this clause has a different name"),
                        ]),
                );
                continue;
            }

            let clause_arity: u8 = clause.params.len().try_into().unwrap();
            if clause_arity != arity {
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message("unterminated function clause")
                        .with_labels(vec![
                            Label::primary(last_clause.source_id(), last_clause.clone())
                                .with_message(
                                "this clause ends with ';', indicating that another clause follows",
                            ),
                            Label::secondary(clause.span.source_id(), clause.span.clone())
                                .with_message("but this clause has a different arity"),
                        ]),
                );
                continue;
            }

            last_clause = clause.span.clone();
        }

        match name {
            None => Ok(Self::Anonymous(AnonymousFun {
                name: None,
                span,
                arity,
                clauses,
            })),
            Some(Name::Var(ident)) => Ok(Self::Recursive(RecursiveFun {
                name: None,
                self_name: ident,
                span,
                arity,
                clauses,
            })),
            Some(Name::Atom(_)) => panic!("funs are not permitted to have non-identifier names"),
        }
    }
}

/// Represents a function name which contains parts which are not yet concrete,
/// i.e. they are expressions which need to be evaluated to know precisely which
/// module or function is referenced
#[derive(Debug, Clone)]
pub struct UnresolvedFunctionName {
    pub span: SourceSpan,
    pub module: Option<Name>,
    pub function: Name,
    pub arity: Arity,
}
impl PartialEq for UnresolvedFunctionName {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module && self.function == other.function && self.arity == other.arity
    }
}
impl Eq for UnresolvedFunctionName {}
impl Hash for UnresolvedFunctionName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module.hash(state);
        self.function.hash(state);
        self.arity.hash(state);
    }
}
impl PartialOrd for UnresolvedFunctionName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.module.partial_cmp(&other.module) {
            None | Some(Ordering::Equal) => match self.function.partial_cmp(&other.function) {
                None | Some(Ordering::Equal) => self.arity.partial_cmp(&other.arity),
                Some(order) => Some(order),
            },
            Some(order) => Some(order),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash)]
pub enum FunctionName {
    Resolved(Spanned<syntax_core::FunctionName>),
    PartiallyResolved(Spanned<syntax_core::FunctionName>),
    Unresolved(UnresolvedFunctionName),
}
impl FunctionName {
    pub fn span(&self) -> SourceSpan {
        match self {
            FunctionName::Resolved(ref spanned) => spanned.span(),
            FunctionName::PartiallyResolved(ref spanned) => spanned.span(),
            FunctionName::Unresolved(UnresolvedFunctionName { span, .. }) => *span,
        }
    }

    pub fn partial_resolution(&self) -> Option<Spanned<syntax_core::FunctionName>> {
        match self {
            Self::PartiallyResolved(partial) => Some(*partial),
            Self::Unresolved(UnresolvedFunctionName {
                span,
                module,
                function,
                arity,
                ..
            }) => match Self::detect(*span, *module, *function, *arity) {
                Self::PartiallyResolved(fun) => Some(fun),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn detect(span: SourceSpan, module: Option<Name>, function: Name, arity: Arity) -> Self {
        match (module, function, arity) {
            (Some(Name::Atom(m)), Name::Atom(f), Arity::Int(a)) => Self::Resolved(Spanned::new(
                span,
                syntax_core::FunctionName::new(m.name, f.name, a),
            )),
            (None, Name::Atom(f), Arity::Int(a)) => Self::PartiallyResolved(Spanned::new(
                span,
                syntax_core::FunctionName::new_local(f.name, a),
            )),
            (m, f, a) => Self::Unresolved(UnresolvedFunctionName {
                span,
                module: m,
                function: f,
                arity: a,
            }),
        }
    }

    pub fn from_clause(clause: &FunctionClause) -> FunctionName {
        match clause {
            &FunctionClause {
                name: Some(ref name),
                span,
                ref params,
                ..
            } => FunctionName::PartiallyResolved(Spanned::new(
                span,
                syntax_core::FunctionName::new_local(
                    name.atom().name,
                    params.len().try_into().unwrap(),
                ),
            )),
            _ => panic!("cannot create a FunctionName from an anonymous FunctionClause!"),
        }
    }
}
impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionName::Resolved(ref spanned) => write!(f, "{}", spanned),
            FunctionName::PartiallyResolved(ref spanned) => write!(f, "{}", spanned),
            FunctionName::Unresolved(UnresolvedFunctionName {
                module: Some(ref module),
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}:{:?}/{:?}", module, function, arity),
            FunctionName::Unresolved(UnresolvedFunctionName {
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}/{:?}", function, arity),
        }
    }
}
