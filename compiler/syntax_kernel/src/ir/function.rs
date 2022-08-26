use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_syntax_base::*;

use super::Expr;

#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub struct Function {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: FunctionName,
    pub vars: Vec<Var>,
    pub body: Box<Expr>,
}
annotated!(Function);
