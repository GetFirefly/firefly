use liblumen_diagnostics::{SourceSpan, Spanned};
use liblumen_syntax_base::*;

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
