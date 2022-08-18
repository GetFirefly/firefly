use std::collections::HashSet;
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
    pub compile: CompileOptions,
    pub on_load: Option<Span<FunctionName>>,
    pub exports: HashSet<Span<FunctionName>>,
    pub nifs: HashSet<Span<FunctionName>>,
    pub functions: Vec<Function>,
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
