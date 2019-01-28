use std::collections::{HashMap, HashSet};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use liblumen_diagnostics::{ByteSpan, Diagnostic, Label};

use super::{ParseError, ParserError};
use super::{Ident, Symbol, Expr, Literal};
use super::{Record, Callback, TypeSpec, TypeDef, TypeSig};
use super::{NamedFunction, FunctionName, ResolvedFunctionName, FunctionClause};
use super::{Attribute, UserAttribute, Deprecation};
use super::{Tuple, Cons, Apply, Remote, Nil};

/// Represents expressions valid at the top level of a module body
#[derive(Debug, Clone, PartialEq)]
pub enum TopLevel {
    Attribute(Attribute),
    Record(Record),
    Function(NamedFunction),
}

/// Represents a complete module, broken down into its constituent parts
///
/// Creating a module via `Module::new` ensures that each field is correctly
/// populated, that sanity checking of the top-level constructs is performed,
/// and that a module is ready for semantic analysis and lowering to IR
///
/// A key step performed by `Module::new` is decorating `ResolvedFunctionName`
/// structs with the current module where appropriate (as this is currently not
/// done during parsing, as the module is constructed last). This means that once
/// constructed, one can use `ResolvedFunctionName` equality in sets/maps, which
/// allows us to easily check definitions, usages, and more.
#[derive(Debug, Clone)]
pub struct Module {
    pub span: ByteSpan,
    pub name: Ident,
    pub vsn: Option<Expr>,
    pub compile: Option<CompileOptions>,
    pub on_load: Option<ResolvedFunctionName>,
    pub imports: HashSet<ResolvedFunctionName>,
    pub exports: HashSet<ResolvedFunctionName>,
    pub types: HashMap<ResolvedFunctionName, TypeDef>,
    pub exported_types: HashSet<ResolvedFunctionName>,
    pub behaviours: HashSet<Ident>,
    pub callbacks: HashMap<ResolvedFunctionName, Callback>,
    pub records: HashMap<Symbol, Record>,
    pub attributes: HashMap<Ident, UserAttribute>,
    pub functions: BTreeMap<ResolvedFunctionName, NamedFunction>,
    // Used for module-level deprecation
    pub deprecation: Option<Deprecation>,
    // Used for function-level deprecation
    pub deprecations: HashSet<Deprecation>,
}
impl Module {
    /// Called by the parser to create the module once all of the top-level expressions have been
    /// parsed, in other words this is the last function called when parsing a module.
    ///
    /// As a result, this function performs some initial linting of the module:
    ///
    /// * If configured to do so, warns if functions are missing type specs
    /// * Warns about type specs for undefined functions
    /// * Warns about redefined attributes
    /// * Errors on invalid syntax in built-in attributes (e.g. -import(..))
    /// * Errors on mismatched function clauses (name/arity)
    /// * Errors on unterminated function clauses
    /// * Errors on redefined functions
    ///
    /// And a few other similar lints
    pub fn new(errs: &mut Vec<ParseError>, span: ByteSpan, name: Ident, mut body: Vec<TopLevel>) -> Self {
        let mut module = Module {
            span,
            name,
            vsn: None,
            on_load: None,
            compile: None,
            imports: HashSet::new(),
            exports: HashSet::new(),
            types: HashMap::new(),
            exported_types: HashSet::new(),
            behaviours: HashSet::new(),
            callbacks: HashMap::new(),
            records: HashMap::new(),
            attributes: HashMap::new(),
            functions: BTreeMap::new(),
            deprecation: None,
            deprecations: HashSet::new(),
        };

        // Functions will be decorated with their type specs as they are added
        // to the module. To accomplish this, we keep track of seen type specs
        // as they are defined, then later look up the spec for a function when
        // a definition is encountered
        let mut specs: HashMap<ResolvedFunctionName, TypeSpec> = HashMap::new();

        // Walk every top-level expression and extend our initial module definition accordingly
        for item in body.drain(..) {
            match item {
                TopLevel::Attribute(Attribute::Vsn(aspan, vsn)) => {
                    if module.vsn.is_none() {
                        module.vsn = Some(vsn);
                        continue;
                    }
                    errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                        Diagnostic::new_error("attribute is already defined")
                            .with_label(Label::new_primary(aspan)
                                .with_message("redefinition occurs here"))
                            .with_label(Label::new_secondary(module.vsn.clone().map(|v| v.span()).unwrap())
                                .with_message("first defined here"))
                    )));
                }
                TopLevel::Attribute(Attribute::OnLoad(aspan, fname)) => {
                    if module.on_load.is_none() {
                        let fname = fname.resolve(module.name.clone());
                        module.on_load = Some(fname);
                        continue;
                    }
                    errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                        Diagnostic::new_error("on_load can only be defined once")
                            .with_label(Label::new_primary(aspan)
                                .with_message("redefinition occurs here"))
                            .with_label(Label::new_secondary(module.on_load.clone().map(|v| v.span).unwrap())
                                .with_message("first defined here"))
                    )))
                }
                TopLevel::Attribute(Attribute::Import(aspan, from_module, mut imports)) => {
                    for import in imports.drain(..) {
                        let import = import.resolve(from_module.clone());
                        match module.imports.get(&import) {
                            None => {
                                module.imports.insert(import);
                            }
                            Some(ResolvedFunctionName { span: ref prev_span, .. }) => {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                    Diagnostic::new_warning("unused import")
                                        .with_label(Label::new_primary(aspan)
                                            .with_message("this import is a duplicate of a previous import"))
                                        .with_label(Label::new_secondary(prev_span.clone())
                                            .with_message("function was first imported here"))
                                )));
                            }
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Export(aspan, mut exports)) => {
                    for export in exports.drain(..) {
                        let export = export.resolve(module.name.clone());
                        match module.exports.get(&export) {
                            None => {
                                module.exports.insert(export);
                            }
                            Some(ResolvedFunctionName { span: ref prev_span, .. }) => {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                    Diagnostic::new_warning("already exported")
                                        .with_label(Label::new_primary(aspan)
                                            .with_message("duplicate export occurs here"))
                                        .with_label(Label::new_secondary(prev_span.clone())
                                            .with_message("function was first exported here"))
                                )));
                            }
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Type(ty)) => {
                    let arity = ty.params.len();
                    let type_name = ResolvedFunctionName {
                        span: ty.name.span.clone(),
                        module: module.name.clone(),
                        function: ty.name.clone(),
                        arity: arity,
                    };
                    match module.types.get(&type_name) {
                        None => {
                            module.types.insert(type_name, ty);
                        }
                        Some(TypeDef { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_warning("type is already defined")
                                    .with_label(Label::new_primary(ty.span)
                                        .with_message("redefinition occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("type was first defined here"))
                            )));
                        }
                    }
                }
                TopLevel::Attribute(Attribute::ExportType(aspan, mut exports)) => {
                    for export in exports.drain(..) {
                        let export = export.resolve(module.name.clone());
                        match module.exported_types.get(&export) {
                            None => {
                                module.exported_types.insert(export);
                            }
                            Some(ResolvedFunctionName { span: ref prev_span, .. }) => {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                    Diagnostic::new_warning("type already exported")
                                        .with_label(Label::new_primary(aspan)
                                            .with_message("duplicate export occurs here"))
                                        .with_label(Label::new_secondary(prev_span.clone())
                                            .with_message("type was first exported here"))
                                )));
                            }
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Behaviour(aspan, b_module)) => {
                    match module.behaviours.get(&b_module) {
                        None => {
                            module.behaviours.insert(b_module);
                        }
                        Some(Ident { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_warning("duplicate behaviour declaration")
                                    .with_label(Label::new_primary(aspan)
                                        .with_message("duplicate declaration occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("first declaration occurs here"))
                            )));
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Callback(callback)) => {
                    let first_sig = callback.sigs.first().unwrap();
                    let arity = first_sig.params.len();

                    // Verify that all clauses match
                    if callback.sigs.len() > 1 {
                        for TypeSig { span: ref sigspan, ref params, .. } in &callback.sigs[1..] {
                            if params.len() != arity {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                    Diagnostic::new_error("mismatched arity")
                                        .with_label(Label::new_primary(sigspan.clone())
                                            .with_message(format!("expected arity of {}", arity)))
                                        .with_label(Label::new_secondary(first_sig.span.clone())
                                            .with_message("expected arity was derived from this clause"))
                                )));
                            }
                        }
                    }
                    // Check for redefinition
                    let cb_name = ResolvedFunctionName {
                        span: callback.span.clone(),
                        module: module.name.clone(),
                        function: callback.function.clone(),
                        arity: arity,
                    };
                    match module.callbacks.get(&cb_name) {
                        None => {
                            module.callbacks.insert(cb_name, callback);
                            continue;
                        }
                        Some(Callback { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_error("cannot redefine callback")
                                    .with_label(Label::new_primary(callback.span)
                                        .with_message("redefinition occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("callback first defined here"))
                            )));
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Spec(typespec)) => {
                    let first_sig = typespec.sigs.first().unwrap();
                    let arity = first_sig.params.len();

                    // Verify that all clauses match
                    if typespec.sigs.len() > 1 {
                        for TypeSig { span: ref sigspan, ref params, .. } in &typespec.sigs[1..] {
                            if params.len() != arity {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                    Diagnostic::new_error("mismatched arity")
                                        .with_label(Label::new_primary(sigspan.clone())
                                            .with_message(format!("expected arity of {}", arity)))
                                        .with_label(Label::new_secondary(first_sig.span.clone())
                                            .with_message("expected arity was derived from this clause"))
                                )));
                            }
                        }
                    }
                    // Check for redefinition
                    let spec_name = ResolvedFunctionName {
                        span: typespec.span.clone(),
                        module: module.name.clone(),
                        function: typespec.function.clone(),
                        arity: arity,
                    };
                    match specs.get(&spec_name) {
                        None => {
                            specs.insert(spec_name.clone(), typespec);
                        }
                        Some(TypeSpec { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_error("spec already defined")
                                    .with_label(Label::new_primary(typespec.span)
                                        .with_message("redefinition occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("spec first defined here"))
                            )));
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Compile(_, compile)) => {
                    match module.compile {
                        None => {
                            let (opts, mut validation_errs) = CompileOptions::from_expr(&module.name, &compile);
                            module.compile = Some(opts);
                            for err in validation_errs.drain(..) {
                                errs.push(to_lalrpop_err!(ParserError::Diagnostic(err)));
                            }
                            continue;
                        }
                        Some(ref mut opts) => {
                            if let Err(mut validation_errs) = opts.merge_from_expr(&module.name, &compile) {
                                for err in validation_errs.drain(..) {
                                    errs.push(to_lalrpop_err!(ParserError::Diagnostic(err)));
                                }
                            }
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Deprecation(mut deprecations)) => {
                    for deprecation in deprecations.drain(..) {
                        match deprecation {
                            Deprecation::Module { span: ref dspan, .. } => {
                                match module.deprecation {
                                    None => {
                                        module.deprecation = Some(deprecation);
                                    }
                                    Some(Deprecation::Module { span: ref orig_span, .. }) => {
                                        errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                            Diagnostic::new_warning("redundant deprecation")
                                                .with_label(Label::new_primary(dspan.clone())
                                                    .with_message("this module is already deprecated by a previous declaration"))
                                                .with_label(Label::new_secondary(orig_span.clone())
                                                    .with_message("deprecation first declared here"))
                                        )));
                                    }
                                    Some(Deprecation::Function { .. }) => unreachable!()
                                }
                            }
                            Deprecation::Function { span: ref fspan, .. } => {
                                if let Some(Deprecation::Module { span: ref mspan, .. }) = module.deprecation {
                                    errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                        Diagnostic::new_warning("redundant deprecation")
                                            .with_label(Label::new_primary(*fspan)
                                                .with_message("module is deprecated, so deprecating functions is redundant"))
                                            .with_label(Label::new_secondary(mspan.clone())
                                                .with_message("module deprecation occurs here"))
                                    )));
                                    continue;
                                }

                                match module.deprecations.get(&deprecation) {
                                    None => {
                                        module.deprecations.insert(deprecation);
                                    }
                                    Some(Deprecation::Function { span: ref prev_span, .. }) => {
                                        errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                            Diagnostic::new_warning("redundant deprecation")
                                                .with_label(Label::new_primary(*fspan)
                                                    .with_message("this function is already deprecated by a previous declaration"))
                                                .with_label(Label::new_secondary(prev_span.clone())
                                                    .with_message("deprecation first declared here"))
                                        )));
                                    }
                                    Some(Deprecation::Module { .. }) => unreachable!()
                                }
                            }
                        }
                    }
                }
                TopLevel::Attribute(Attribute::Custom(attr)) => {
                    match attr.name.name.as_str().get() {
                        "module" => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_error("multiple module declarations")
                                    .with_label(Label::new_primary(attr.span.clone())
                                        .with_message("invalid declaration occurs here"))
                                    .with_label(Label::new_secondary(module.name.span.clone())
                                        .with_message("module first declared here"))
                            )));
                            continue;
                        }
                        "dialyzer" => {
                            // Drop dialyzer attributes as they are unused
                            continue;
                        }
                        _ => ()
                    }
                    match module.attributes.get(&attr.name) {
                        None => {
                            module.attributes.insert(attr.name.clone(), attr);
                        }
                        Some(UserAttribute { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_warning("redefined attribute")
                                    .with_label(Label::new_primary(attr.span.clone())
                                        .with_message("redefinition occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("previously defined here"))
                            )));
                            module.attributes.insert(attr.name.clone(), attr);
                        }
                    }
                }
                TopLevel::Record(record) => {
                    // TODO: Normalize records so that all fields have explicit initializers
                    //   * Use `undefined` for fields that have no default value
                    // TODO: Check for duplicate fields
                    let name = record.name.name.clone();
                    match module.records.get(&name) {
                        None => {
                            module.records.insert(name, record);
                        }
                        Some(Record { span: ref prev_span, .. }) => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_error("record already defined")
                                    .with_label(Label::new_primary(record.span.clone())
                                        .with_message("duplicate definition occurs here"))
                                    .with_label(Label::new_secondary(prev_span.clone())
                                        .with_message("previously defined here"))
                            )));
                        }
                    }
                }
                TopLevel::Function(mut function @ NamedFunction { .. }) => {
                    let name = &function.name;
                    let resolved_name = ResolvedFunctionName {
                        span: name.span.clone(),
                        module: module.name.clone(),
                        function: name.clone(),
                        arity: function.arity,
                    };
                    let warn_missing_specs = module.compile
                        .as_ref()
                        .map(|c| c.warn_missing_spec)
                        .unwrap_or(false);
                    function.spec = match specs.get(&resolved_name) {
                        None if warn_missing_specs => {
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_warning("missing function spec")
                                    .with_label(Label::new_primary(function.span.clone())
                                        .with_message("expected type spec for this function"))
                            )));
                            None
                        }
                        None => None,
                        Some(spec) => Some(spec.clone()),
                    };
                    match module.functions.entry(resolved_name) {
                        Entry::Vacant(f) => {
                            f.insert(function);
                        }
                        Entry::Occupied(initial_def) => {
                            let def = initial_def.into_mut();
                            errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                                Diagnostic::new_error("clauses from the same function should be grouped together")
                                    .with_label(Label::new_primary(function.span.clone())
                                        .with_message("found more clauses here"))
                                    .with_label(Label::new_secondary(def.span.clone())
                                        .with_message("function is first defined here"))
                            )));
                            def.clauses.append(&mut function.clauses);
                        }
                    }
                }
            }
        }

        // Ensure internal pseudo-locals are defined
        module.define_pseudolocals();

        // Verify on_load function exists
        if let Some(ref on_load_name) = module.on_load {
            if !module.functions.contains_key(on_load_name) {
                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_error("invalid on_load function")
                        .with_label(Label::new_primary(on_load_name.span.clone())
                            .with_message("this function is not defined in this module"))
                )));
            }
        }

        // Check for orphaned type specs
        for (spec_name, spec) in &specs {
            if !module.functions.contains_key(spec_name) {
                errs.push(to_lalrpop_err!(ParserError::Diagnostic(
                    Diagnostic::new_warning("type spec for undefined function")
                        .with_label(Label::new_primary(spec.span.clone())
                            .with_message("this type spec has no corresponding function definition"))
                )));
            }
        }

        module
    }

    // Every module in Erlang has some functions implicitly defined for internal use:
    //
    // * `module_info/0` (exported)
    // * `module_info/1` (exported)
    // * `record_info/2`
    // * `behaviour_info/1` (optional)
    fn define_pseudolocals(&mut self) {
        let mod_info_0 = fun!(module_info () ->
            apply!(remote!(erlang, get_module_info), Expr::Literal(Literal::Atom(self.name.clone())))
        );
        let mod_info_1 = fun!(module_info (Key) ->
            apply!(remote!(erlang, get_module_info), Expr::Literal(Literal::Atom(self.name.clone())), var!(Key))
        );

        self.define_function(mod_info_0);
        self.define_function(mod_info_1);

        if self.callbacks.len() > 0 {
            let callbacks = self.callbacks.iter().fold(nil!(), |acc, (cbname, cb)| {
                if cb.optional {
                    acc
                } else {
                    cons!(tuple!(atom_from_sym!(cbname.function.name.clone()), int!(cbname.arity as i64)), acc)
                }
            });
            let opt_callbacks = self.callbacks.iter().fold(nil!(), |acc, (cbname, cb)| {
                if cb.optional {
                    cons!(tuple!(atom_from_sym!(cbname.function.name.clone()), int!(cbname.arity as i64)), acc)
                } else {
                    acc
                }
            });

            let behaviour_info_1 = fun!(behaviour_info
                                        (atom!(callbacks)) -> callbacks;
                                        (atom!(optional_callbacks)) -> opt_callbacks);

            self.define_function(behaviour_info_1);
        }
    }

    fn define_function(&mut self, f: NamedFunction) {
        let name = ResolvedFunctionName {
            span: f.span.clone(),
            module: self.name.clone(),
            function: f.name.clone(),
            arity: f.arity,
        };
        self.functions.insert(name, f);
    }
}
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
    // Treats all warnings as errors
    pub warnings_as_errors: bool,
    // Disables warnings
    pub no_warn: bool,
    // Exports all functions
    pub export_all: bool,
    // Prevents auto importing any functions
    pub no_auto_import: bool,
    // Prevents auto importing the specified functions
    pub no_auto_imports: HashSet<ResolvedFunctionName>,
    // Warns if export_all is used
    pub warn_export_all: bool,
    // Warns when exported variables are used
    pub warn_export_vars: bool,
    // Warns when variables are shadowed
    pub warn_shadow_vars: bool,
    // Warns when a function is unused
    pub warn_unused_function: bool,
    // Disables the unused function warning for the specified functions
    pub no_warn_unused_functions: HashSet<ResolvedFunctionName>,
    // Warns about unused imports
    pub warn_unused_import: bool,
    // Warns about unused variables
    pub warn_unused_var: bool,
    // Warns about unused records
    pub warn_unused_record: bool,
    // Warns about missing type specs
    pub warn_missing_spec: bool,
}
impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            compile_info: HashMap::new(),
            file: None,
            warnings_as_errors: false,
            no_warn: false,
            export_all: false,
            no_auto_import: false,
            no_auto_imports: HashSet::new(),
            warn_export_all: true,
            warn_export_vars: true,
            warn_shadow_vars: true,
            warn_unused_function: true,
            no_warn_unused_functions: HashSet::new(),
            warn_unused_import: true,
            warn_unused_var: true,
            warn_unused_record: true,
            warn_missing_spec: false,
        }
    }
}
impl CompileOptions {
    pub fn from_expr(module: &Ident, expr: &Expr) -> (Self, Vec<Diagnostic>) {
        let mut opts = CompileOptions::default();
        match opts.merge_from_expr(module, expr) {
            Ok(_) =>
                (opts, Vec::new()),
            Err(errs) =>
                (opts, errs),
        }
    }

    pub fn merge_from_expr(&mut self, module: &Ident, expr: &Expr) -> Result<(), Vec<Diagnostic>> {
        self.set_option(module, expr)
    }

    fn set_option(&mut self, module: &Ident, expr: &Expr) -> Result<(), Vec<Diagnostic>> {
        let mut diagnostics = Vec::new();
        match expr {
            // e.g. -compile(export_all).
            &Expr::Literal(Literal::Atom(ref option_name)) => {
                match option_name.as_str().get() {
                    "export_all" =>
                        self.export_all = true,
                    "nowarn_export_all" =>
                        self.warn_export_all = false,
                    "nowarn_shadow_vars" =>
                        self.warn_shadow_vars = false,
                    "nowarn_unused_function" =>
                        self.warn_unused_function = false,
                    "nowarn_unused_vars" =>
                        self.warn_unused_var = false,
                    "no_auto_import" =>
                        self.no_auto_import = true,
                    _name => {
                        diagnostics.push(
                            Diagnostic::new_warning("invalid compile option")
                                .with_label(Label::new_primary(option_name.span)
                                    .with_message("this option is either unsupported or unrecognized"))
                        );
                    }
                }
            }
            // e.g. -compile([export_all, nowarn_unused_function]).
            &Expr::Cons(Cons { ref head, ref tail, .. }) => {
                self.compiler_opts_from_list(&mut diagnostics, module, to_list(head, tail))
            }
            // e.g. -compile({nowarn_unused_function, [some_fun/0]}).
            &Expr::Tuple(Tuple { ref elements, .. }) => {
                if let Some((head, tail)) = elements.split_first() {
                    if let &Expr::Literal(Literal::Atom(ref option_name)) = head {
                        match option_name.as_str().get() {
                            "no_auto_import" => {
                                self.no_auto_imports(&mut diagnostics, module, tail);
                            }
                            "nowarn_unused_function" => {
                                self.no_warn_unused_functions(&mut diagnostics, module, tail);
                            }
                            _name => {
                                diagnostics.push(
                                    Diagnostic::new_warning("invalid compile option")
                                        .with_label(Label::new_primary(option_name.span)
                                            .with_message("this option is either unsupported or unrecognized"))
                                );
                            }
                        }
                    }
                }
            }
            term => {
                diagnostics.push(
                    Diagnostic::new_warning("invalid compile option")
                        .with_label(Label::new_primary(term.span())
                            .with_message("unexpected expression: expected atom, list, or tuple"))
                );
            }
        }

        if diagnostics.len() > 0 {
            return Err(diagnostics);
        }

        Ok(())
    }


    fn compiler_opts_from_list(&mut self, diagnostics: &mut Vec<Diagnostic>, module: &Ident, options: Vec<Expr>) {
        for option in options {
            match self.set_option(module, &option) {
                Ok(_) => continue,
                Err(mut diags) => diagnostics.append(&mut diags),
            }
        }
    }

    fn no_auto_imports(&mut self, diagnostics: &mut Vec<Diagnostic>, module: &Ident, imports: &[Expr]) {
        for import in imports {
            match import {
                Expr::FunctionName(FunctionName::PartiallyResolved(name)) => {
                    self.no_auto_imports.insert(name.resolve(module.clone()));
                }
                other => {
                    diagnostics.push(
                        Diagnostic::new_warning("invalid compile option")
                            .with_label(Label::new_primary(other.span())
                                .with_message("expected function name/arity term for no_auto_imports"))
                    );
                }
            }
        }
    }

    fn no_warn_unused_functions(&mut self, diagnostics: &mut Vec<Diagnostic>, module: &Ident, funs: &[Expr]) {
        for fun in funs {
            match fun {
                Expr::FunctionName(FunctionName::PartiallyResolved(name)) => {
                    self.no_warn_unused_functions.insert(name.resolve(module.clone()));
                }
                other => {
                    diagnostics.push(
                        Diagnostic::new_warning("invalid compile option")
                            .with_label(Label::new_primary(other.span())
                                .with_message("expected function name/arity term for no_warn_unused_functions"))
                    );
                }
            }
        }
    }
}

fn to_list(head: &Expr, tail: &Expr) -> Vec<Expr> {
    let mut list = Vec::new();
    match head {
        &Expr::Cons(Cons { head: ref head2, tail: ref tail2, .. }) => {
            let mut h = to_list(head2, tail2);
            list.append(&mut h);
        }
        expr =>
            list.push(expr.clone()),
    }
    match tail {
        &Expr::Cons(Cons { head: ref head2, tail: ref tail2, .. }) => {
            let mut t = to_list(head2, tail2);
            list.append(&mut t);
        }
        &Expr::Nil(_) =>
            (),
        expr =>
            list.push(expr.clone()),
    }

    list
}
