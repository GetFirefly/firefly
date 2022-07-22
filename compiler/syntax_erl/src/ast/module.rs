use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};

use anyhow::anyhow;

use liblumen_diagnostics::{Diagnostic, Label, Reporter, SourceSpan, Span, Spanned};
use liblumen_syntax_core as syntax_core;
use liblumen_util::emit::Emit;

use super::*;

/// Represents expressions valid at the top level of a module body
#[derive(Debug, Clone, PartialEq, Spanned)]
pub enum TopLevel {
    Module(Ident),
    Attribute(Attribute),
    Record(Record),
    Function(Function),
}
impl TopLevel {
    pub fn module_name(&self) -> Option<Ident> {
        match self {
            Self::Module(name) => Some(*name),
            _ => None,
        }
    }

    pub fn is_attribute(&self) -> bool {
        match self {
            Self::Attribute(_) => true,
            _ => false,
        }
    }

    pub fn is_record(&self) -> bool {
        match self {
            Self::Record(_) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function(_) => true,
            _ => false,
        }
    }
}

/// Represents a complete module, broken down into its constituent parts
///
/// Creating a module via `Module::new` ensures that each field is correctly
/// populated, that sanity checking of the top-level constructs is performed,
/// and that a module is ready for semantic analysis and lowering to IR
///
/// A key step performed by `Module::new` is decorating `FunctionName`
/// structs with the current module where appropriate (as this is currently not
/// done during parsing, as the module is constructed last). This means that once
/// constructed, one can use `FunctionName` equality in sets/maps, which
/// allows us to easily check definitions, usages, and more.
#[derive(Debug, Clone, Spanned)]
pub struct Module {
    #[span]
    pub span: SourceSpan,
    pub name: Ident,
    pub vsn: Option<Expr>,
    pub author: Option<Expr>,
    pub compile: Option<CompileOptions>,
    pub on_load: Option<Span<syntax_core::FunctionName>>,
    pub nifs: HashSet<Span<syntax_core::FunctionName>>,
    pub imports: HashMap<syntax_core::FunctionName, Span<syntax_core::Signature>>,
    pub exports: HashSet<Span<syntax_core::FunctionName>>,
    pub removed: HashMap<syntax_core::FunctionName, (SourceSpan, Ident)>,
    pub types: HashMap<syntax_core::FunctionName, TypeDef>,
    pub exported_types: HashSet<Span<syntax_core::FunctionName>>,
    pub specs: HashMap<syntax_core::FunctionName, TypeSpec>,
    pub behaviours: HashSet<Ident>,
    pub callbacks: HashMap<syntax_core::FunctionName, Callback>,
    pub records: HashMap<Symbol, Record>,
    pub attributes: HashMap<Ident, UserAttribute>,
    pub functions: BTreeMap<syntax_core::FunctionName, Function>,
    // Used for module-level deprecation
    pub deprecation: Option<Deprecation>,
    // Used for function-level deprecation
    pub deprecations: HashSet<Deprecation>,
}
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("ast")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;
        write!(f, "{:#?}", self)?;
        Ok(())
    }
}
impl Module {
    pub fn name(&self) -> Symbol {
        self.name.name
    }

    pub fn record(&self, name: Symbol) -> Option<&Record> {
        self.records.get(&name)
    }

    pub fn is_local(&self, name: &syntax_core::FunctionName) -> bool {
        let local_name = name.to_local();
        self.functions.contains_key(&local_name)
    }

    pub fn is_import(&self, name: &syntax_core::FunctionName) -> bool {
        let local_name = name.to_local();
        !self.is_local(&local_name) && self.imports.contains_key(&local_name)
    }

    /// Creates a new, empty module with the given name and span
    pub fn new(name: Ident, span: SourceSpan) -> Self {
        Self {
            span,
            name,
            vsn: None,
            author: None,
            on_load: None,
            nifs: HashSet::new(),
            compile: None,
            imports: HashMap::new(),
            exports: HashSet::new(),
            removed: HashMap::new(),
            types: HashMap::new(),
            exported_types: HashSet::new(),
            specs: HashMap::new(),
            behaviours: HashSet::new(),
            callbacks: HashMap::new(),
            records: HashMap::new(),
            attributes: HashMap::new(),
            functions: BTreeMap::new(),
            deprecation: None,
            deprecations: HashSet::new(),
        }
    }

    /// Called by the parser for Erlang Abstract Format, which relies on us detecting the module name in the given forms
    pub fn new_from_pp(
        reporter: &Reporter,
        span: SourceSpan,
        body: Vec<TopLevel>,
    ) -> anyhow::Result<Self> {
        let name = body.iter().find_map(|t| t.module_name()).ok_or_else(|| {
            anyhow!("invalid module, no module declaration present in given forms")
        })?;
        Ok(Self::new_with_forms(reporter, span, name, body))
    }

    /// Called by the parser to create the module once all of the top-level expressions have been
    /// parsed, in other words this is the last function called when parsing a module.
    ///
    /// As a result, this function performs initial semantic analysis of the module.
    ///
    pub fn new_with_forms(
        reporter: &Reporter,
        span: SourceSpan,
        name: Ident,
        body: Vec<TopLevel>,
    ) -> Self {
        use crate::passes::{CanonicalizeSyntax, SemanticAnalysis};
        use liblumen_pass::Pass;

        let mut passes = SemanticAnalysis::new(reporter.clone(), span, name)
            .chain(CanonicalizeSyntax::new(reporter.clone()));
        match passes.run(body) {
            Ok(module) => module,
            Err(reason) => {
                let reason = reason.to_string();
                reporter.diagnostic(
                    Diagnostic::error()
                        .with_message(&reason)
                        .with_labels(vec![Label::primary(name.span.source_id(), name.span)
                            .with_message("error occurred while compiling this module")]),
                );
                Self::new(name, span)
            }
        }
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Module) -> bool {
        if self.name != other.name {
            return false;
        }
        if self.vsn != other.vsn {
            return false;
        }
        if self.on_load != other.on_load {
            return false;
        }
        if self.nifs != other.nifs {
            return false;
        }
        if self.imports != other.imports {
            return false;
        }
        if self.exports != other.exports {
            return false;
        }
        if self.types != other.types {
            return false;
        }
        if self.exported_types != other.exported_types {
            return false;
        }
        if self.behaviours != other.behaviours {
            return false;
        }
        if self.callbacks != other.callbacks {
            return false;
        }
        if self.records != other.records {
            return false;
        }
        if self.attributes != other.attributes {
            return false;
        }
        if self.functions != other.functions {
            return false;
        }
        true
    }
}

/// This structure holds all module-specific compiler options
/// and configuration; it is passed through all phases of
/// compilation and is a superset of options in CompilerSettings
/// where applicable
#[derive(Debug, Clone)]
pub struct CompileOptions {
    // Same as erlc, prints informational warnings about
    // binary matching optimizations
    pub compile_info: HashMap<Symbol, Expr>,
    // Used to override the filename used in errors/warnings
    pub file: Option<String>,
    pub verbose: bool,
    pub report_errors: bool,
    pub report_warnings: bool,
    // Treats all warnings as errors
    pub warnings_as_errors: bool,
    // Disables warnings
    pub no_warn: bool,
    // Exports all functions
    pub export_all: bool,
    // Prevents auto importing any functions
    pub no_auto_import: bool,
    // Prevents auto importing the specified functions
    pub no_auto_imports: HashSet<syntax_core::FunctionName>,
    // Warns if export_all is used
    pub warn_export_all: bool,
    // Warns when exported variables are used
    pub warn_export_vars: bool,
    // Warns when variables are shadowed
    pub warn_shadow_vars: bool,
    // Warns when a function is unused
    pub warn_unused_function: bool,
    // Disables the unused function warning for the specified functions
    pub no_warn_unused_functions: HashSet<Span<syntax_core::FunctionName>>,
    // Warns about unused imports
    pub warn_unused_import: bool,
    // Warns about unused variables
    pub warn_unused_var: bool,
    // Warns about unused records
    pub warn_unused_record: bool,
    pub warn_untyped_record: bool,
    pub warn_unused_type: bool,
    pub warn_removed: bool,
    pub warn_nif_inline: bool,
    pub warn_bif_clash: bool,
    // Warns about missing type specs
    pub warn_missing_spec: bool,
    pub warn_missing_spec_all: bool,
    pub warn_deprecated_function: bool,
    pub warn_deprecated_type: bool,
    pub warn_obsolete_guard: bool,
    pub inline: bool,
    // Inlines the given functions
    pub inline_functions: HashSet<Span<syntax_core::FunctionName>>,
}
impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            compile_info: HashMap::new(),
            file: None,
            verbose: true,
            report_errors: true,
            report_warnings: true,
            warnings_as_errors: false,
            no_warn: false,
            export_all: false,
            no_auto_import: false,
            no_auto_imports: HashSet::new(),
            inline: false,
            inline_functions: HashSet::new(),

            // Warning toggles
            warn_export_all: true,
            warn_export_vars: true,
            warn_shadow_vars: true,
            warn_unused_function: true,
            no_warn_unused_functions: HashSet::new(),
            warn_unused_import: true,
            warn_unused_var: true,
            warn_unused_record: true,
            warn_untyped_record: false,
            warn_unused_type: true,
            warn_removed: true,
            warn_nif_inline: true,
            warn_bif_clash: true,
            warn_missing_spec: false,
            warn_missing_spec_all: false,
            warn_deprecated_function: true,
            warn_deprecated_type: true,
            warn_obsolete_guard: true,
        }
    }
}
impl CompileOptions {
    pub fn from_expr(module: Ident, expr: &Expr, reporter: &Reporter) -> Result<Self, Self> {
        let mut opts = CompileOptions::default();
        match opts.merge_from_expr(module, expr, reporter) {
            Ok(_) => Ok(opts),
            Err(_) => Err(opts),
        }
    }

    pub(crate) fn merge_from_expr(
        &mut self,
        module: Ident,
        expr: &Expr,
        reporter: &Reporter,
    ) -> Result<(), ()> {
        self.set_option(module, expr, reporter)
    }

    fn set_option(&mut self, module: Ident, expr: &Expr, reporter: &Reporter) -> Result<(), ()> {
        match expr {
            // e.g. -compile(export_all).
            &Expr::Literal(Literal::Atom(ref option_name)) => {
                match option_name.as_str().get() {
                    "no_native" => (), // Disables hipe compilation, not relevant for us
                    "inline" => self.inline = true,

                    "export_all" => self.export_all = true,

                    "no_auto_import" => self.no_auto_import = true,

                    "report_errors" => self.report_errors = true,
                    "report_warnings" => self.report_errors = true,
                    "verbose" => self.verbose = true,

                    "inline_list_funcs" => {
                        let funs = [
                            ("lists", "all", 2),
                            ("lists", "any", 2),
                            ("lists", "foreach", 2),
                            ("lists", "map", 2),
                            ("lists", "flatmap", 2),
                            ("lists", "filter", 2),
                            ("lists", "foldl", 3),
                            ("lists", "foldr", 3),
                            ("lists", "mapfoldl", 3),
                            ("lists", "mapfoldr", 3),
                        ];
                        for (m, f, a) in funs.iter() {
                            self.inline_functions.insert(Span::new(
                                option_name.span,
                                syntax_core::FunctionName::new(
                                    Symbol::intern(m),
                                    Symbol::intern(f),
                                    *a,
                                ),
                            ));
                        }
                    }

                    // Warning toggles
                    "warn_export_all" => self.warn_export_all = true,
                    "nowarn_export_all" => self.warn_export_all = false,

                    "warn_shadow_vars" => self.warn_shadow_vars = true,
                    "nowarn_shadow_vars" => self.warn_shadow_vars = false,

                    "warn_unused_function" => self.warn_unused_function = true,
                    "nowarn_unused_function" => self.warn_unused_function = false,

                    "warn_unused_import" => self.warn_unused_import = true,
                    "nowarn_unused_import" => self.warn_unused_import = false,

                    "warn_unused_type" => self.warn_unused_type = true,
                    "nowarn_unused_type" => self.warn_unused_type = false,

                    "warn_export_vars" => self.warn_export_vars = true,
                    "nowarn_export_vars" => self.warn_export_vars = false,

                    "warn_unused_vars" => self.warn_unused_var = true,
                    "nowarn_unused_vars" => self.warn_unused_var = false,

                    "warn_bif_clash" => self.warn_bif_clash = true,
                    "nowarn_bif_clash" => self.warn_bif_clash = false,

                    "warn_unused_record" => self.warn_unused_record = true,
                    "nowarn_unused_record" => self.warn_unused_record = false,

                    "warn_deprecated_function" => self.warn_deprecated_function = true,
                    "nowarn_deprecated_function" => self.warn_deprecated_function = false,

                    "warn_deprecated_type" => self.warn_deprecated_type = true,
                    "nowarn_deprecated_type" => self.warn_deprecated_type = false,

                    "warn_obsolete_guard" => self.warn_obsolete_guard = true,
                    "nowarn_obsolete_guard" => self.warn_obsolete_guard = false,

                    "warn_untyped_record" => self.warn_untyped_record = true,
                    "nowarn_untyped_record" => self.warn_untyped_record = false,

                    "warn_missing_spec" => self.warn_missing_spec = true,
                    "nowarn_missing_spec" => self.warn_missing_spec = false,

                    "warn_missing_spec_all" => self.warn_missing_spec_all = true,
                    "nowarn_missing_spec_all" => self.warn_missing_spec_all = false,

                    "warn_removed" => self.warn_removed = true,
                    "nowarn_removed" => self.warn_removed = false,

                    "warn_nif_inline" => self.warn_nif_inline = true,
                    "nowarn_nif_inline" => self.warn_nif_inline = false,

                    _name => {
                        reporter.diagnostic(
                            Diagnostic::warning()
                                .with_message("invalid compile option")
                                .with_labels(vec![Label::primary(
                                    option_name.span.source_id(),
                                    option_name.span,
                                )
                                .with_message(
                                    "this option is either unsupported or unrecognized",
                                )]),
                        );
                        return Err(());
                    }
                }
            }
            // e.g. -compile([export_all, nowarn_unused_function]).
            &Expr::Cons(Cons {
                ref head, ref tail, ..
            }) => self.compiler_opts_from_list(module, to_list(head, tail), reporter),
            // e.g. -compile({nowarn_unused_function, [some_fun/0]}).
            &Expr::Tuple(Tuple { ref elements, .. }) if elements.len() == 2 => {
                if let &Expr::Literal(Literal::Atom(ref option_name)) = &elements[0] {
                    let list = to_list_simple(&elements[1]);
                    match option_name.as_str().get() {
                        "no_auto_import" => self.no_auto_imports(module, &list, reporter),
                        "nowarn_unused_function" => {
                            self.no_warn_unused_functions(module, &list, reporter)
                        }
                        "inline" => self.inline_functions(module, &list, reporter),
                        // Ignored
                        "hipe" => {}
                        _name => {
                            reporter.diagnostic(
                                Diagnostic::warning()
                                    .with_message("invalid compile option")
                                    .with_labels(vec![Label::primary(
                                        option_name.span.source_id(),
                                        option_name.span,
                                    )
                                    .with_message(
                                        "this option is either unsupported or unrecognized",
                                    )]),
                            );
                            return Err(());
                        }
                    }
                }
            }
            term => {
                let term_span = term.span();
                reporter.diagnostic(
                    Diagnostic::warning()
                        .with_message("invalid compile option")
                        .with_labels(vec![Label::primary(term_span.source_id(), term_span)
                            .with_message(
                                "unexpected expression: expected atom, list, or tuple",
                            )]),
                );
                return Err(());
            }
        }

        Ok(())
    }

    fn compiler_opts_from_list(&mut self, module: Ident, options: Vec<Expr>, reporter: &Reporter) {
        for option in options.iter() {
            let _ = self.set_option(module, option, reporter);
        }
    }

    fn no_auto_imports(&mut self, module: Ident, imports: &[Expr], reporter: &Reporter) {
        for import in imports {
            match import {
                Expr::FunctionName(FunctionName::PartiallyResolved(name)) => {
                    self.no_auto_imports.insert(name.resolve(module.name));
                }
                Expr::Tuple(tup) if tup.elements.len() == 2 => {
                    match (&tup.elements[0], &tup.elements[1]) {
                        (
                            Expr::Literal(Literal::Atom(name)),
                            Expr::Literal(Literal::Integer(_, arity)),
                        ) => {
                            let name = syntax_core::FunctionName::new(
                                module.name,
                                name.name,
                                arity.to_arity(),
                            );
                            self.no_auto_imports.insert(name);
                            continue;
                        }
                        _ => (),
                    }
                }
                other => {
                    let other_span = other.span();
                    reporter.diagnostic(
                        Diagnostic::warning()
                            .with_message("invalid compile option")
                            .with_labels(vec![Label::primary(other_span.source_id(), other_span)
                                .with_message(
                                    "expected function name/arity term for no_auto_imports",
                                )]),
                    );
                }
            }
        }
    }

    fn no_warn_unused_functions(&mut self, _module: Ident, funs: &[Expr], reporter: &Reporter) {
        for fun in funs {
            match fun {
                Expr::FunctionName(FunctionName::PartiallyResolved(name)) => {
                    self.no_warn_unused_functions.insert(*name);
                }
                other => {
                    let other_span = other.span();
                    reporter.diagnostic(
                        Diagnostic::warning()
                            .with_message("invalid compile option")
                            .with_labels(vec![Label::primary(other_span.source_id(), other_span)
                                .with_message(
                                "expected function name/arity term for no_warn_unused_functions",
                            )]),
                    );
                }
            }
        }
    }

    fn inline_functions(&mut self, module: Ident, funs: &[Expr], reporter: &Reporter) {
        for fun in funs {
            match fun {
                Expr::FunctionName(FunctionName::PartiallyResolved(name)) => {
                    let name = Span::new(name.span(), name.resolve(module.name));
                    self.inline_functions.insert(name);
                    continue;
                }
                Expr::Tuple(tup) if tup.elements.len() == 2 => {
                    match (&tup.elements[0], &tup.elements[1]) {
                        (
                            Expr::Literal(Literal::Atom(name)),
                            Expr::Literal(Literal::Integer(_, arity)),
                        ) => {
                            let name = Span::new(
                                tup.span,
                                syntax_core::FunctionName::new(
                                    module.name,
                                    name.name,
                                    arity.to_arity(),
                                ),
                            );
                            self.inline_functions.insert(name);
                            continue;
                        }
                        _ => (),
                    }
                }
                _ => (),
            }

            let fun_span = fun.span();
            reporter.diagnostic(
                Diagnostic::warning()
                    .with_message("invalid compile option")
                    .with_labels(vec![Label::primary(fun_span.source_id(), fun_span)
                        .with_message("expected function name/arity term for inline")]),
            );
        }
    }
}

fn to_list_simple(mut expr: &Expr) -> Vec<Expr> {
    let mut list = Vec::new();
    loop {
        match expr {
            Expr::Cons(cons) => {
                list.push((*cons.head).clone());
                expr = &cons.tail;
            }
            Expr::Literal(Literal::Nil(_)) => {
                return list;
            }
            _ => {
                list.push(expr.clone());
                return list;
            }
        }
    }
}

fn to_list(head: &Expr, tail: &Expr) -> Vec<Expr> {
    let mut list = Vec::new();
    match head {
        &Expr::Cons(Cons {
            head: ref head2,
            tail: ref tail2,
            ..
        }) => {
            let mut h = to_list(head2, tail2);
            list.append(&mut h);
        }
        expr => list.push(expr.clone()),
    }
    match tail {
        &Expr::Cons(Cons {
            head: ref head2,
            tail: ref tail2,
            ..
        }) => {
            let mut t = to_list(head2, tail2);
            list.append(&mut t);
        }
        &Expr::Literal(Literal::Nil(_)) => (),
        expr => list.push(expr.clone()),
    }

    list
}
