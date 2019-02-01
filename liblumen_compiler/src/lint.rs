#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use failure::Error;

use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};
use liblumen_syntax::ast::*;
use liblumen_syntax::visitor::ImmutableVisitor;
use liblumen_syntax::{Ident, Symbol};

use super::compiler::Compiler;

pub fn module(_compiler: &mut Compiler, module: &Module) -> Result<(), Error> {
    let opts = module
        .compile
        .clone()
        .unwrap_or_else(|| CompileOptions::default());
    let mut linter = Linter::new(module, opts);
    linter.lint(module)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Variable {
    name: Ident,
    exported: bool,
    bound: bool,
    used: bool,
    notsafe: bool,
}
impl Variable {}

pub struct Linter<'ast> {
    module: &'ast Module,
    opts: CompileOptions,
    clashes: Vec<ResolvedFunctionName>,
    called: HashMap<ResolvedFunctionName, Vec<ByteSpan>>,
    //usage_calls: HashMap<ResolvedFunctionName, FunctionName>,
    usage_imported: HashSet<ResolvedFunctionName>,
    usage_records: HashSet<Ident>,
    usage_types: HashMap<ResolvedFunctionName, Vec<ByteSpan>>,
    error: Option<Error>,
}
impl<'ast> Linter<'ast> {
    fn new(module: &'ast Module, opts: CompileOptions) -> Self {
        Linter {
            module,
            opts,
            clashes: Vec::new(),
            called: HashMap::new(),
            usage_imported: HashSet::new(),
            usage_records: HashSet::new(),
            usage_types: HashMap::new(),
            error: None,
        }
    }

    fn lint(&mut self, module: &'ast Module) -> Result<(), Error> {
        self.visit(module);

        let err = std::mem::replace(&mut self.error, None);
        match err {
            None => Ok(()),
            Some(err) => Err(err),
        }
    }
}

/// Tasks
///
/// * Determine if imports clash with imported BIFs
/// * Track function calls
///   * Imports
///   * Un-exported locals
///   * Deprecated functions
/// * Track usages of local records
/// * Determine if behaviours are implemented fully
/// * Determine if behaviours clash
/// * Check for records without type info (optional warning)
/// * Check for undefined module references
/// * Check for shadowed variables in patterns
/// * Check for invalid aliasing of binaries and maps
///   * e.g. `<<A:8>> = <<B:4,C:4>>` or `<<A:8>> = <<A:8>>`
///   * e.g. `#{foo => A} = #{foo => bar}.` (`A` is unbound)
/// * Validate bit size expressions/patterns
/// * Ensure guards are not shadowed by locals
/// * Ensure record fields exist when referenced
/// * Normalize record construction so that all fields are initialized
/// * Check format strings for `io:format/fwrite` and `io_lib:format/fwrite`
impl<'ast> ImmutableVisitor<'ast> for Linter<'ast> {
    fn visit(&mut self, _module: &'ast Module) {}
}
