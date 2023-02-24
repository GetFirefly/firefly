use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use firefly_intern::Symbol;
use firefly_number::Int;
use firefly_syntax_base::FunctionName;
use firefly_util::diagnostics::*;

use super::{Arity, Clause, Expr, Ident, Literal, Name, TypeSpec, Var};
use crate::ParserError;

/// A top-level function definition
#[derive(Debug, Clone, Spanned)]
pub struct Function {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub arity: u8,
    pub spec: Option<TypeSpec>,
    pub is_nif: bool,
    pub clauses: Vec<(Option<Name>, Clause)>,
    pub var_counter: usize,
    pub fun_counter: usize,
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
    pub fn local_name(&self) -> FunctionName {
        FunctionName::new_local(self.name.name, self.arity)
    }

    pub fn next_var(&mut self, span: Option<SourceSpan>) -> Ident {
        let id = self.var_counter;
        self.var_counter += 1;
        let var = format!("${}", id);
        let mut ident = Ident::from_str(&var);
        match span {
            None => ident,
            Some(span) => {
                ident.span = span;
                ident
            }
        }
    }

    pub fn new(
        diagnostics: &DiagnosticsHandler,
        span: SourceSpan,
        clauses: Vec<(Option<Name>, Clause)>,
    ) -> Result<Self, ParserError> {
        debug_assert!(clauses.len() > 0);

        let head_pair = &clauses[0];
        let head_name = head_pair.0.as_ref();
        let head = &head_pair.1;
        let head_span = head.span.clone();
        let name = head_name.unwrap().atom();
        let arity = head.patterns.len().try_into().unwrap();

        // Check clauses
        let mut last_clause = head_span.clone();
        for (clause_name, clause) in clauses.iter().skip(1) {
            if clause_name.is_none() {
                return Err(ParserError::ShowDiagnostic {
                    diagnostic: diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("expected named function clause")
                        .with_primary_label(
                            clause.span,
                            "this clause has no name, but a name is required here",
                        )
                        .with_secondary_label(
                            last_clause,
                            "expected a clause with the same name as this clause",
                        )
                        .take(),
                });
            }

            let clause_name = clause_name.clone().unwrap();
            let clause_arity: u8 = clause.patterns.len().try_into().unwrap();

            if clause_name != Name::Atom(name) {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("unterminated function clause")
                    .with_primary_label(
                        last_clause,
                        "this clause ends with ';', indicating that another clause follows",
                    )
                    .with_secondary_label(clause.span, "but this clause has a different name")
                    .emit();
                continue;
            }
            if clause_arity != arity {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("unterminated function clause")
                    .with_primary_label(
                        last_clause,
                        "this clause ends with ';', indicating that another clause follows",
                    )
                    .with_secondary_label(clause.span, "but this clause has a different arity")
                    .emit();
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
            var_counter: 0,
            fun_counter: 0,
        })
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct Guard {
    #[span]
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
#[derive(Debug, Clone, Spanned)]
pub struct RecursiveFun {
    #[span]
    pub span: SourceSpan,
    // Name is only set when an anonymous function is assigned a name by a compiler pass
    // Immediately after parsing, it is always None
    pub name: Option<Ident>,
    // The self_name is the name bound to the function within its body, which allows the function to call itself
    pub self_name: Ident,
    pub arity: u8,
    pub clauses: Vec<(Name, Clause)>,
}
impl PartialEq for RecursiveFun {
    fn eq(&self, other: &Self) -> bool {
        self.self_name == other.self_name
            && self.arity == other.arity
            && self.clauses == other.clauses
    }
}

/// An anonymous function that cannot refer to itself within its body
#[derive(Debug, Clone, Spanned)]
pub struct AnonymousFun {
    #[span]
    pub span: SourceSpan,
    // Name is only set when an anonymous function is assigned a name by a compiler pass
    // Immediately after parsing, it is always None
    pub name: Option<Ident>,
    pub arity: u8,
    pub clauses: Vec<Clause>,
}
impl PartialEq for AnonymousFun {
    fn eq(&self, other: &Self) -> bool {
        self.arity == other.arity && self.clauses == other.clauses
    }
}

#[derive(Debug, Clone, PartialEq, Spanned)]
pub enum Fun {
    Recursive(RecursiveFun),
    Anonymous(AnonymousFun),
}
impl Fun {
    pub fn new(
        diagnostics: &DiagnosticsHandler,
        span: SourceSpan,
        mut clauses: Vec<(Option<Name>, Clause)>,
    ) -> Result<Self, ParserError> {
        debug_assert!(clauses.len() > 0);
        let head_clause = &clauses[0];
        let name = head_clause.0.clone();
        let head = &head_clause.1;
        let head_span = head.span.clone();
        let arity = head.patterns.len().try_into().unwrap();

        // Check clauses
        let mut last_clause = head_span.clone();
        for (clause_name, clause) in clauses.iter().skip(1) {
            if name.is_some() && clause_name.is_none() {
                return Err(ParserError::ShowDiagnostic {
                    diagnostic: diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("expected named function clause")
                        .with_primary_label(
                            clause.span,
                            "this clause has no name, but a name is required here",
                        )
                        .with_secondary_label(
                            last_clause,
                            "expected a clause with the same name as this clause",
                        )
                        .take(),
                });
            }

            if name.is_none() && clause_name.is_some() {
                return Err(ParserError::ShowDiagnostic {
                    diagnostic: diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("mismatched function clause")
                        .with_primary_label(clause.span, "this clause is named")
                        .with_secondary_label(
                            last_clause,
                            "but this clause is unnamed, all clauses must share the same name",
                        )
                        .take(),
                });
            }

            if *clause_name != name {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("unterminated function clause")
                    .with_primary_label(
                        last_clause,
                        "this clause ends with ';', indicating that another clause follows",
                    )
                    .with_secondary_label(clause.span, "but this clause has a different name")
                    .emit();
                continue;
            }

            let clause_arity: u8 = clause.patterns.len().try_into().unwrap();
            if clause_arity != arity {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("unterminated function clause")
                    .with_primary_label(
                        last_clause,
                        "this clause ends with ';', indicating that another clause follows",
                    )
                    .with_secondary_label(clause.span, "but this clause has a different arity")
                    .emit();
                continue;
            }

            last_clause = clause.span.clone();
        }

        match name {
            None => Ok(Self::Anonymous(AnonymousFun {
                name: None,
                span,
                arity,
                clauses: clauses.drain(..).map(|(_, c)| c).collect(),
            })),
            Some(Name::Var(ident)) => Ok(Self::Recursive(RecursiveFun {
                name: None,
                self_name: ident,
                span,
                arity,
                clauses: clauses.drain(..).map(|(n, c)| (n.unwrap(), c)).collect(),
            })),
            Some(Name::Atom(_)) => panic!("funs are not permitted to have non-identifier names"),
        }
    }
}

/// Represents a function name which contains parts which are not yet concrete,
/// i.e. they are expressions which need to be evaluated to know precisely which
/// module or function is referenced
#[derive(Debug, Clone, Spanned)]
pub struct UnresolvedFunctionName {
    #[span]
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

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash, Spanned)]
pub enum FunctionVar {
    Resolved(Span<FunctionName>),
    PartiallyResolved(Span<FunctionName>),
    Unresolved(UnresolvedFunctionName),
}
impl From<Span<FunctionName>> for FunctionVar {
    fn from(name: Span<FunctionName>) -> Self {
        if name.is_local() {
            Self::PartiallyResolved(name)
        } else {
            Self::Resolved(name)
        }
    }
}
impl FunctionVar {
    pub fn module(&self) -> Option<Symbol> {
        match self {
            Self::Resolved(mfa) => mfa.module,
            Self::PartiallyResolved(_) => None,
            Self::Unresolved(name) => match name.module {
                Some(Name::Atom(f)) => Some(f.name),
                _ => None,
            },
        }
    }

    pub fn function(&self) -> Option<Symbol> {
        match self {
            Self::Resolved(mfa) => Some(mfa.function),
            Self::PartiallyResolved(mfa) => Some(mfa.function),
            Self::Unresolved(name) => match name.function {
                Name::Atom(f) => Some(f.name),
                _ => None,
            },
        }
    }

    pub fn mfa(&self) -> (Option<Expr>, Expr, Expr) {
        match self {
            Self::Resolved(name) | Self::PartiallyResolved(name) => {
                let span = name.span();
                let module = name
                    .module
                    .map(|m| Expr::Literal(Literal::Atom(Ident::new(m, span))));
                let function = Expr::Literal(Literal::Atom(Ident::new(name.function, span)));
                let arity = Expr::Literal(Literal::Integer(span, (name.arity as i64).into()));
                (module, function, arity)
            }
            Self::Unresolved(UnresolvedFunctionName {
                span,
                module,
                function,
                arity,
            }) => {
                let module = module.map(|m| match m {
                    Name::Var(id) => Expr::Var(Var(id)),
                    Name::Atom(id) => Expr::Literal(Literal::Atom(id)),
                });
                let function = match function {
                    Name::Var(id) => Expr::Var(Var(*id)),
                    Name::Atom(id) => Expr::Literal(Literal::Atom(*id)),
                };
                let arity = match arity {
                    Arity::Int(i) => Expr::Literal(Literal::Integer(*span, Int::Small(*i as i64))),
                    Arity::Var(id) => Expr::Var(Var(*id)),
                };
                (module, function, arity)
            }
        }
    }

    pub fn new(span: SourceSpan, module: Symbol, function: Symbol, arity: u8) -> Self {
        Self::Resolved(Span::new(span, FunctionName::new(module, function, arity)))
    }

    pub fn new_local(span: SourceSpan, function: Symbol, arity: u8) -> Self {
        Self::PartiallyResolved(Span::new(span, FunctionName::new_local(function, arity)))
    }

    pub fn partial_resolution(&self) -> Option<Span<FunctionName>> {
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
            (Some(Name::Atom(m)), Name::Atom(f), Arity::Int(a)) => {
                Self::new(span, m.name, f.name, a)
            }
            (None, Name::Atom(f), Arity::Int(a)) => Self::new_local(span, f.name, a),
            (m, f, a) => Self::Unresolved(UnresolvedFunctionName {
                span,
                module: m,
                function: f,
                arity: a,
            }),
        }
    }
}
impl fmt::Display for FunctionVar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Resolved(ref spanned) => write!(f, "{}", spanned),
            Self::PartiallyResolved(ref spanned) => write!(f, "{}", spanned),
            Self::Unresolved(UnresolvedFunctionName {
                module: Some(ref module),
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}:{:?}/{:?}", module, function, arity),
            Self::Unresolved(UnresolvedFunctionName {
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}/{:?}", function, arity),
        }
    }
}
