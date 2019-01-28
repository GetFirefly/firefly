use std::fmt;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};

use crate::preprocessor::PreprocessorError;

use super::{TryParseResult, ParseError, ParserError};
use super::{Ident, Name, Expr, TypeSpec};

/// Represents a fully-resolved function name, with module/function/arity explicit
#[derive(Debug, Clone)]
pub struct ResolvedFunctionName {
    pub span: ByteSpan,
    pub module: Ident,
    pub function: Ident,
    pub arity: usize,
}
impl PartialEq for ResolvedFunctionName {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module &&
        self.function == other.function &&
        self.arity == other.arity
    }
}
impl Eq for ResolvedFunctionName {}
impl Hash for ResolvedFunctionName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module.hash(state);
        self.function.hash(state);
        self.arity.hash(state);
    }
}
impl PartialOrd for ResolvedFunctionName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (xm, xf, xa) = (self.module, self.function, self.arity);
        let (ym, yf, ya) = (other.module, other.function, other.arity);
        match xm.partial_cmp(&ym) {
            None | Some(Ordering::Equal) => match xf.partial_cmp(&yf) {
                None | Some(Ordering::Equal) => xa.partial_cmp(&ya),
                Some(order) => Some(order),
            },
            Some(order) => Some(order)
        }
    }
}
impl Ord for ResolvedFunctionName {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Represents a partially-resolved function name, not yet associated with a module
/// This is typically used to express local captures, e.g. `fun do_stuff/0`
#[derive(Debug, Clone)]
pub struct PartiallyResolvedFunctionName {
    pub span: ByteSpan,
    pub function: Ident,
    pub arity: usize,
}
impl PartiallyResolvedFunctionName {
    pub fn resolve(&self, module: Ident) -> ResolvedFunctionName {
        ResolvedFunctionName {
            span: self.span.clone(),
            module,
            function: self.function.clone(),
            arity: self.arity,
        }
    }
}
impl PartialEq for PartiallyResolvedFunctionName {
    fn eq(&self, other: &Self) -> bool {
        self.function == other.function &&
        self.arity == other.arity
    }
}
impl Eq for PartiallyResolvedFunctionName {}
impl Hash for PartiallyResolvedFunctionName {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.function.hash(state);
        self.arity.hash(state);
    }
}
impl PartialOrd for PartiallyResolvedFunctionName {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (xf, xa) = (self.function, self.arity);
        let (yf, ya) = (other.function, other.arity);
        match xf.partial_cmp(&yf) {
            None | Some(Ordering::Equal) => xa.partial_cmp(&ya),
            Some(order) => Some(order)
        }
    }
}

/// Represents a function name which contains parts which are not yet concrete,
/// i.e. they are expressions which need to be evaluated to know precisely which
/// module or function is referenced
#[derive(Debug, Clone)]
pub struct UnresolvedFunctionName {
    pub span: ByteSpan,
    pub module: Option<Name>,
    pub function: Name,
    pub arity: usize,
}
impl PartialEq for UnresolvedFunctionName {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module &&
        self.function == other.function &&
        self.arity == other.arity
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
            None | Some(Ordering::Equal) =>
                match self.function.partial_cmp(&other.function) {
                    None | Some(Ordering::Equal) =>
                        self.arity.partial_cmp(&other.arity),
                    Some(order) =>
                        Some(order),
                },
            Some(order) => Some(order)
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash)]
pub enum FunctionName {
    Resolved(ResolvedFunctionName),
    PartiallyResolved(PartiallyResolvedFunctionName),
    Unresolved(UnresolvedFunctionName),
}
impl FunctionName {
    pub fn span(&self) -> ByteSpan {
        match self {
            &FunctionName::Resolved(ResolvedFunctionName { ref span, .. }) => span.clone(),
            &FunctionName::PartiallyResolved(PartiallyResolvedFunctionName { ref span, .. }) => span.clone(),
            &FunctionName::Unresolved(UnresolvedFunctionName { ref span, .. }) => span.clone(),
        }
    }

    pub fn detect(span: ByteSpan, module: Option<Name>, function: Name, arity: usize) -> Self {
        if module.is_none() {
            return match function {
                Name::Atom(f) => FunctionName::PartiallyResolved(PartiallyResolvedFunctionName {
                    span,
                    function: f,
                    arity: arity,
                }),
                Name::Var(_) => FunctionName::Unresolved(UnresolvedFunctionName {
                    span,
                    module: None,
                    function,
                    arity,
                }),
            };
        }

        if let Some(Name::Atom(m)) = module {
            if let Name::Atom(f) = function {
                return FunctionName::Resolved(ResolvedFunctionName {
                    span,
                    module: m,
                    function: f,
                    arity,
                });
            }
        }

        FunctionName::Unresolved(UnresolvedFunctionName {
            span,
            module,
            function,
            arity: arity,
        })
    }

    pub fn from_clause(clause: &FunctionClause) -> FunctionName {
        match clause {
            &FunctionClause {
                name: Some(ref name),
                ref span,
                ref params,
                ..
            } => {
                FunctionName::PartiallyResolved(PartiallyResolvedFunctionName {
                    span: span.clone(),
                    function: name.clone(),
                    arity: params.len(),
                })
            }
            _ => {
                panic!("cannot create a FunctionName from an anonymous FunctionClause!")
            }
        }
    }
}
impl fmt::Display for FunctionName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FunctionName::Resolved(ResolvedFunctionName {
                ref module,
                ref function,
                arity,
                ..
            }) => write!(f, "{}:{}/{}", module, function, arity),
            FunctionName::PartiallyResolved(PartiallyResolvedFunctionName {
                ref function,
                arity,
                ..
            }) => write!(f, "{}/{}", function, arity),
            FunctionName::Unresolved(UnresolvedFunctionName {
                module: Some(ref module),
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}:{:?}/{}", module, function, arity),
            FunctionName::Unresolved(UnresolvedFunctionName {
                ref function,
                arity,
                ..
            }) => write!(f, "{:?}/{}", function, arity),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NamedFunction {
    pub span: ByteSpan,
    pub name: Ident,
    pub arity: usize,
    pub clauses: Vec<FunctionClause>,
    pub spec: Option<TypeSpec>,
}
impl PartialEq for NamedFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name &&
        self.arity == other.arity &&
        self.clauses == other.clauses &&
        self.spec == other.spec
    }
}
impl NamedFunction {
    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, clauses: Vec<FunctionClause>) -> TryParseResult<Self> {
        debug_assert!(clauses.len() > 0);
        let (head, rest) = clauses
            .split_first()
            .unwrap();

        if head.name.is_none() {
            return Err(to_lalrpop_err!(PreprocessorError::Diagnostic(
                Diagnostic::new_error("expected named function")
                    .with_label(Label::new_primary(head.span)
                        .with_message("this clause has no name, but a name is required here"))
            )));
        }

        let head_span = &head.span;
        let name = head.name.clone().unwrap();
        let params = &head.params;
        let arity = params.len();

        // Check clauses
        let mut last_clause = head_span.clone();
        for clause in rest.iter() {
            if clause.name.is_none() {
                return Err(to_lalrpop_err!(PreprocessorError::Diagnostic(
                    Diagnostic::new_error("expected named function clause")
                        .with_label(Label::new_primary(clause.span)
                            .with_message("this clause has no name, but a name is required here"))
                        .with_label(Label::new_secondary(last_clause)
                            .with_message("expected a clause with the same name as this clause"))
                )));
            }

            let clause_span = &clause.span;
            let clause_name = clause.name.clone().unwrap();
            let clause_params = &clause.params;
            let clause_arity = clause_params.len();

            if clause_name != name {
                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("unterminated function clause")
                        .with_label(Label::new_primary(last_clause.clone())
                            .with_message("this clause ends with ';', indicating that another clause follows"))
                        .with_label(Label::new_secondary(clause_span.clone())
                            .with_message("but this clause has a different name"))
                )));
                continue;
            }
            if clause_arity != arity {
                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("unterminated function clause")
                        .with_label(Label::new_primary(last_clause.clone())
                            .with_message("this clause ends with ';', indicating that another clause follows"))
                        .with_label(Label::new_secondary(clause_span.clone())
                            .with_message("but this clause has a different arity"))
                )));
                continue;
            }

            last_clause = clause_span.clone();
        }

        Ok(NamedFunction {
            span,
            name: name.clone(),
            arity,
            clauses,
            spec: None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Lambda {
    pub span: ByteSpan,
    pub arity: usize,
    pub clauses: Vec<FunctionClause>,
}
impl PartialEq for Lambda {
    fn eq(&self, other: &Self) -> bool {
        self.arity == other.arity &&
        self.clauses == other.clauses
    }
}
impl Lambda {
    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, clauses: Vec<FunctionClause>) -> TryParseResult<Self> {
        debug_assert!(clauses.len() > 0);
        let (head, rest) = clauses
            .split_first()
            .unwrap();

        let head_span = &head.span;
        let params = &head.params;
        let arity = params.len();

        // Check clauses
        let mut last_clause = head_span.clone();
        for clause in rest.iter() {
            let clause_span = &clause.span;
            let clause_name = &clause.name;
            let clause_params = &clause.params;
            let clause_arity = clause_params.len();

            if clause_name.is_some() {
                return Err(to_lalrpop_err!(PreprocessorError::Diagnostic(
                    Diagnostic::new_error("mismatched function clause")
                        .with_label(Label::new_primary(clause_span.clone())
                            .with_message("this clause is named"))
                        .with_label(Label::new_secondary(last_clause.clone())
                            .with_message("but this clause is unnamed, all clauses must share the same name"))
                )))
            }

            if clause_arity != arity {
                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("mismatched function clause")
                        .with_label(Label::new_primary(clause_span.clone())
                            .with_message("the arity of this clause does not match the previous clause"))
                        .with_label(Label::new_secondary(last_clause.clone())
                            .with_message("this is the previous clause"))
                )));
                continue;
            }

            last_clause = clause_span.clone();
        }

        Ok(Lambda {
            span,
            arity,
            clauses,
        })
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum Function {
    Named(NamedFunction),
    Unnamed(Lambda),
}
impl Function {
    pub fn span(&self) -> ByteSpan {
        match self {
            &Function::Named(NamedFunction { ref span, .. }) =>
                span.clone(),
            &Function::Unnamed(Lambda { ref span, .. }) =>
                span.clone(),
        }
    }

    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, clauses: Vec<FunctionClause>) -> TryParseResult<Self> {
        debug_assert!(clauses.len() > 0);
        let (head, _rest) = clauses
            .split_first()
            .unwrap();

        if head.name.is_some() {
            Ok(Function::Named(NamedFunction::new(errs, span, clauses)?))
        } else {
            Ok(Function::Unnamed(Lambda::new(errs, span, clauses)?))
        }
    }
}

#[derive(Debug, Clone)]
pub struct FunctionClause {
    pub span: ByteSpan,
    pub name: Option<Ident>,
    pub params: Vec<Expr>,
    pub guard: Option<Vec<Guard>>,
    pub body: Vec<Expr>,
}
impl PartialEq for FunctionClause {
    fn eq(&self, other: &FunctionClause) -> bool {
        self.name == other.name &&
        self.params == other.params &&
        self.guard == other.guard &&
        self.body == other.body
    }
}
impl FunctionClause {
    pub fn new(
        span: ByteSpan,
        name: Option<Ident>,
        params: Vec<Expr>,
        guard: Option<Vec<Guard>>,
        body: Vec<Expr>,
    ) -> Self {
        FunctionClause {
            span,
            name,
            params,
            guard,
            body
        }
    }
}

#[derive(Debug, Clone)]
pub struct Guard {
    pub span: ByteSpan,
    pub conditions: Vec<Expr>,
}
impl PartialEq for Guard {
    fn eq(&self, other: &Guard) -> bool {
        self.conditions == other.conditions
    }
}
