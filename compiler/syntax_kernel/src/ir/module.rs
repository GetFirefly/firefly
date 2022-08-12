use std::collections::{HashMap, HashSet};
use std::fmt;

use liblumen_diagnostics::{SourceSpan, Span, Spanned};
use liblumen_intern::Ident;
use liblumen_syntax_base::*;
use liblumen_util::emit::Emit;

use crate::printer::PrettyPrinter;

use super::*;

#[derive(Debug, Clone, Spanned, PartialEq, Eq)]
pub struct Module {
    #[span]
    pub span: SourceSpan,
    pub annotations: Annotations,
    pub name: Ident,
    pub functions: Vec<Function>,
    pub exports: HashSet<Span<FunctionName>>,
    pub imports: HashMap<FunctionName, Span<Signature>>,
    pub attributes: HashMap<Ident, Expr>,
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
        Some("kernel")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;

        write!(f, "{}", self)?;
        Ok(())
    }
}
