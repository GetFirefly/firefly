use std::fmt;

use liblumen_diagnostics::{SourceSpan, Spanned};
use liblumen_syntax_base::*;
use liblumen_syntax_core as syntax_core;

use super::*;
use crate::BiMap;

#[derive(Debug, Clone)]
pub struct IValues {
    pub annotations: Annotations,
    pub values: Vec<Expr>,
}
impl IValues {
    #[inline]
    pub fn new(values: Vec<Expr>) -> Self {
        Self {
            annotations: Annotations::default(),
            values,
        }
    }
}
annotated!(IValues);
impl Spanned for IValues {
    fn span(&self) -> SourceSpan {
        self.values[0].span()
    }
}
impl Eq for IValues {}
impl PartialEq for IValues {
    fn eq(&self, other: &Self) -> bool {
        self.values.eq(&other.values)
    }
}
impl PartialOrd for IValues {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IValues {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.values.cmp(&other.values)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IFun {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub vars: Vec<Var>,
    pub body: Box<Expr>,
}
annotated!(IFun);
impl Eq for IFun {}
impl PartialEq for IFun {
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars && self.body == other.body
    }
}
impl PartialOrd for IFun {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IFun {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.vars
            .cmp(&other.vars)
            .then_with(|| self.body.cmp(&other.body))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ISet {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub vars: Vec<Var>,
    pub arg: Box<Expr>,
    pub body: Option<Box<Expr>>,
}
annotated!(ISet);
impl Into<Expr> for ISet {
    #[inline]
    fn into(self) -> Expr {
        Expr::Set(self)
    }
}
impl ISet {
    pub fn new(span: SourceSpan, vars: Vec<Var>, arg: Expr, body: Option<Expr>) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            vars,
            arg: Box::new(arg),
            body: body.map(Box::new),
        }
    }
}
impl Eq for ISet {}
impl PartialEq for ISet {
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars && self.arg == other.arg && self.body == other.body
    }
}
impl PartialOrd for ISet {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ISet {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.vars
            .cmp(&other.vars)
            .then_with(|| self.arg.cmp(&other.arg))
            .then_with(|| self.body.cmp(&other.body))
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct ILetRec {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub defs: Vec<(Var, IFun)>,
}
annotated!(ILetRec);
impl Eq for ILetRec {}
impl PartialEq for ILetRec {
    fn eq(&self, other: &Self) -> bool {
        self.defs == other.defs
    }
}
impl PartialOrd for ILetRec {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ILetRec {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.defs.cmp(&other.defs)
    }
}

#[derive(Debug, Clone, Spanned)]
pub struct IAlias {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub vars: Vec<Var>,
    pub pattern: Box<Expr>,
}
annotated!(IAlias);
impl IAlias {
    pub fn new(span: SourceSpan, vars: Vec<Var>, pattern: Expr) -> Self {
        Self {
            span,
            annotations: Annotations::default(),
            vars,
            pattern: Box::new(pattern),
        }
    }
}
impl Eq for IAlias {}
impl PartialEq for IAlias {
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars && self.pattern == other.pattern
    }
}
impl PartialOrd for IAlias {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IAlias {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.vars
            .cmp(&other.vars)
            .then_with(|| self.pattern.cmp(&other.pattern))
    }
}

#[derive(Clone, Spanned)]
pub struct IClause {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub isub: BiMap,
    pub osub: BiMap,
    pub patterns: Vec<Expr>,
    pub guard: Option<Box<syntax_core::Expr>>,
    pub body: Box<syntax_core::Expr>,
}
annotated!(IClause);
impl IClause {
    pub fn arg(&self) -> &Expr {
        self.patterns.first().unwrap()
    }

    pub fn match_type(&self) -> MatchType {
        self.arg().match_type()
    }

    pub fn is_var_clause(&self) -> bool {
        match self.match_type() {
            MatchType::Var => true,
            _ => false,
        }
    }
}
impl fmt::Debug for IClause {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("IClause")
            .field("span", &self.span)
            .field("annotations", &self.annotations)
            .field("patterns", &self.patterns)
            .field("guard", &self.guard)
            .field("body", &self.body)
            .finish()
    }
}
impl Eq for IClause {}
impl PartialEq for IClause {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns && self.guard == other.guard && self.body == other.body
    }
}
impl PartialOrd for IClause {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IClause {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.patterns.cmp(&other.patterns)
    }
}
