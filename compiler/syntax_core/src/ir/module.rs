use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_intern::Ident;
use firefly_util::emit::Emit;

use super::*;
use crate::printer::PrettyPrinter;

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
    pub attributes: HashMap<Ident, Expr>,
    pub functions: BTreeMap<FunctionName, Function>,
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
        Some("core")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;

        write!(f, "{}", self)?;
        Ok(())
    }
}
