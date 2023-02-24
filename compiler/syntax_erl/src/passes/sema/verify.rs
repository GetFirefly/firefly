use core::ops::ControlFlow;
use std::collections::{BTreeMap, BTreeSet};

use anyhow::anyhow;

use firefly_intern::Symbol;
use firefly_pass::Pass;
use firefly_syntax_base::{ApplicationMetadata, Deprecation, FunctionName};
use firefly_util::diagnostics::*;

use crate::ast::*;
use crate::visit::{self, VisitMut};

/// Verifies that all declared exports have matching definitions
pub struct VerifyExports<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
impl<'p> VerifyExports<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
impl<'p> Pass for VerifyExports<'p> {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        use core::cell::OnceCell;

        // Only calculate similar functions if we have an invalid export, which should be rare
        let similar_functions = OnceCell::new();

        for export in module.exports.iter() {
            if !module.functions.contains_key(export.as_ref()) {
                // We need to calculate similar functions, so populate the set now
                let similar = similar_functions.get_or_init(|| {
                    let mut similar = Vec::new();
                    for (name, function) in module.functions.iter() {
                        similar.push(Span::new(function.span, name.to_string()));
                    }
                    similar
                });

                let name = export.to_string();
                let most_similar = similar
                    .iter()
                    .map(|f| (strsim::jaro_winkler(&name, &f).abs(), f))
                    .max_by(|(x_score, _), (ref y_score, _)| x_score.total_cmp(y_score))
                    .and_then(|(score, f)| if score < 0.85 { None } else { Some(f) });

                match most_similar {
                    None => {
                        let span = export.span();
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid export")
                            .with_primary_label(
                                span,
                                "the referenced function is not defined in this module",
                            )
                            .emit();
                    }
                    Some(f) => {
                        let span = export.span();
                        let msg = format!("maybe you meant to export {} instead?", &f);
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("invalid export")
                            .with_primary_label(
                                span,
                                "the referenced function is not defined in this module",
                            )
                            .with_secondary_label(f.span(), msg)
                            .emit();
                    }
                }
            }
        }

        Ok(module)
    }
}

/// Verifies that the on_load function exists, if -on_load is present
pub struct VerifyOnLoadFunctions<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
impl<'p> VerifyOnLoadFunctions<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
impl<'p> Pass for VerifyOnLoadFunctions<'p> {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        if let Some(on_load_name) = module.on_load.as_ref() {
            if !module.functions.contains_key(on_load_name.as_ref()) {
                let span = on_load_name.span();
                self.diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid on_load function")
                    .with_primary_label(span, "this function is not defined in this module")
                    .emit();
            }
        }

        Ok(module)
    }
}

/// Like `VerifyExports`, but for `-nifs`; ensures all NIF declarations have a corresponding definition.
pub struct VerifyNifs<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
impl<'p> VerifyNifs<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
impl<'p> Pass for VerifyNifs<'p> {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        for nif in module.nifs.iter() {
            match module.functions.get(nif.as_ref()) {
                None => {
                    let span = nif.span();
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid -nif declaration")
                        .with_primary_label(
                            span,
                            "the referenced function is not defined in this module",
                        )
                        .emit();
                }
                Some(fun) => {
                    if !fun.is_nif {
                        let span = fun.span;
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("misplaced -nif declaration")
                            .with_primary_label(
                                span,
                                "expected -nif declaration to precede the function it references",
                            )
                            .emit();
                    }
                }
            }
        }

        Ok(module)
    }
}

/// Verifies that all declared type specs are associated with a function definition
pub struct VerifyTypeSpecs<'p> {
    diagnostics: &'p DiagnosticsHandler,
}
impl<'p> VerifyTypeSpecs<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler) -> Self {
        Self { diagnostics }
    }
}
impl<'p> Pass for VerifyTypeSpecs<'p> {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        for (spec_name, spec) in module.specs.iter() {
            let local_spec_name = spec_name.to_local();
            if !module.functions.contains_key(&local_spec_name) {
                self.diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("type spec for undefined function")
                    .with_primary_label(
                        spec.span,
                        "this type spec has no corresponding function definition",
                    )
                    .emit();
            }
        }
        Ok(module)
    }
}

/// Verifies that the callee of local function calls is defined or imported, or is dynamic and thus not statically analyzable
///
/// Additionally, checks if the callee is known to be deprecated and raises appropriate diagnostics.
///
/// NOTE: We could extend this analysis to cover calls to other modules, since at the point this analysis is run, we have
/// access to the entire set of modules that was provided to the compiler, however this does not account for cases in which
/// we're only compiling a library and thus only a subset of the modules is known - we could make such analysis optional and
/// only perform it when the full set of modules is known.
pub struct VerifyCalls<'p> {
    diagnostics: &'p DiagnosticsHandler,
    app: &'p ApplicationMetadata,
}
impl<'p> VerifyCalls<'p> {
    pub fn new(diagnostics: &'p DiagnosticsHandler, app: &'p ApplicationMetadata) -> Self {
        Self { diagnostics, app }
    }
}
impl<'p> Pass for VerifyCalls<'p> {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let module_name = module.name.name;
        let locals = module.functions.keys().copied().collect::<BTreeSet<_>>();
        let imports = module
            .imports
            .iter()
            .map(|(name, sig)| (*name, sig.mfa()))
            .collect::<BTreeMap<FunctionName, FunctionName>>();

        let mut has_errors = false;
        for (_, function) in module.functions.iter_mut() {
            let mut visitor = VerifyCallsVisitor {
                diagnostics: self.diagnostics,
                app: self.app,
                module: module_name,
                locals: &locals,
                imports: &imports,
                has_errors: false,
            };
            visitor.visit_mut_function(function);
            if visitor.has_errors {
                has_errors = true;
            }
        }

        if has_errors {
            Err(anyhow!(
                "semantic analysis found one or more errors, see diagnostics for details"
            ))
        } else {
            Ok(module)
        }
    }
}

struct VerifyCallsVisitor<'a> {
    diagnostics: &'a DiagnosticsHandler,
    app: &'a ApplicationMetadata,
    module: Symbol,
    locals: &'a BTreeSet<FunctionName>,
    imports: &'a BTreeMap<FunctionName, FunctionName>,
    has_errors: bool,
}
impl<'a> VisitMut<()> for VerifyCallsVisitor<'a> {
    fn visit_mut_apply(&mut self, apply: &mut Apply) -> ControlFlow<()> {
        for arg in apply.args.iter_mut() {
            let _ = visit::visit_mut_expr(self, arg);
        }
        let span = apply.span();
        let arity = apply.args.len() as u8;
        match apply.callee.as_ref() {
            Expr::Remote(Remote {
                span: rspan,
                module,
                function,
                ..
            }) => match (module.as_atom(), function.as_atom()) {
                (Some(m), Some(f)) if m.name == self.module => {
                    let name = FunctionName::new_local(f.name, arity);
                    if !self.locals.contains(&name) {
                        let message =
                            format!("the function {} is not defined in this module", &name);
                        self.has_errors = true;
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("reference to undefined function")
                            .with_primary_label(*rspan, message)
                            .emit();
                    }
                    ControlFlow::Continue(())
                }
                (Some(m), Some(f)) => {
                    let name = FunctionName::new(m.name, f.name, arity);
                    match self.app.get_function_deprecation(&name) {
                        None => ControlFlow::Continue(()),
                        Some(Deprecation::Module { span: dspan, flag }) => {
                            let note = format!("this module will be deprecated {}", &flag);
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("use of deprecated module")
                                .with_primary_label(m.span, note)
                                .with_secondary_label(dspan, "deprecated here")
                                .emit();
                            ControlFlow::Continue(())
                        }
                        Some(Deprecation::Function {
                            span: dspan, flag, ..
                        }) => {
                            let note = format!("this function will be deprecated {}", &flag);
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("use of deprecated function")
                                .with_primary_label(f.span, note)
                                .with_secondary_label(dspan, "deprecated here")
                                .emit();
                            ControlFlow::Continue(())
                        }
                        // These deprecation types have all been converted to Deprecation::Function
                        Some(Deprecation::FunctionAnyArity { .. }) => unreachable!(),
                    }
                }
                (None, Some(f)) => {
                    let name = FunctionName::new_local(f.name, arity);
                    if !self.locals.contains(&name) {
                        match self.imports.get(&name) {
                            None => {
                                let message = format!(
                                    "the function {} is not defined or imported in this module",
                                    &name
                                );
                                self.has_errors = true;
                                self.diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message("reference to undefined function")
                                    .with_primary_label(f.span, message)
                                    .emit();
                            }
                            Some(imported) => match self.app.get_function_deprecation(&imported) {
                                None => (),
                                Some(Deprecation::Module { span: dspan, flag }) => {
                                    let note =
                                        format!("this function will be deprecated {}", &flag);
                                    self.diagnostics
                                        .diagnostic(Severity::Warning)
                                        .with_message("use of deprecated module")
                                        .with_primary_label(f.span, note)
                                        .with_secondary_label(dspan, "deprecated here")
                                        .emit();
                                }
                                Some(Deprecation::Function {
                                    span: dspan, flag, ..
                                }) => {
                                    let note =
                                        format!("this function will be deprecated {}", &flag);
                                    self.diagnostics
                                        .diagnostic(Severity::Warning)
                                        .with_message("use of deprecated function")
                                        .with_primary_label(f.span, note)
                                        .with_secondary_label(dspan, "deprecated here")
                                        .emit();
                                }
                                // These deprecation types have all been converted to Deprecation::Function
                                Some(Deprecation::FunctionAnyArity { .. }) => unreachable!(),
                            },
                        }
                    }
                    ControlFlow::Continue(())
                }
                _ => ControlFlow::Continue(()),
            },
            Expr::FunctionVar(FunctionVar::Resolved(name)) => {
                if name.module == Some(self.module) {
                    let local_name = name.item.to_local();
                    if !self.locals.contains(&local_name) {
                        let message =
                            format!("the function {} is not defined in this module", &local_name);
                        self.has_errors = true;
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("reference to undefined function")
                            .with_primary_label(name.span(), message)
                            .emit();
                    }
                } else {
                    match self.app.get_function_deprecation(&name) {
                        None => (),
                        Some(Deprecation::Module { span: dspan, flag }) => {
                            let note = format!("this function will be deprecated {}", &flag);
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("use of deprecated module")
                                .with_primary_label(name.span(), note)
                                .with_secondary_label(dspan, "deprecated here")
                                .emit();
                        }
                        Some(Deprecation::Function {
                            span: dspan, flag, ..
                        }) => {
                            let note = format!("this function will be deprecated {}", &flag);
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("use of deprecated function")
                                .with_primary_label(name.span(), note)
                                .with_secondary_label(dspan, "deprecated here")
                                .emit();
                        }
                        // These deprecation types have all been converted to Deprecation::Function
                        Some(Deprecation::FunctionAnyArity { .. }) => unreachable!(),
                    }
                }
                if name.arity > arity {
                    let message = format!(
                        "{} requires {} arguments, but only {} were provided",
                        name, name.arity, arity
                    );
                    self.has_errors = true;
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("missing arguments")
                        .with_primary_label(span, message)
                        .emit();
                } else if name.arity < arity {
                    let message = format!(
                        "{} only takes {} arguments, but {} were provided",
                        name, name.arity, arity
                    );
                    self.has_errors = true;
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("too many arguments")
                        .with_primary_label(span, message)
                        .emit();
                }
                ControlFlow::Continue(())
            }
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                let local_name = FunctionName::new_local(name.function, arity);
                if !self.locals.contains(&local_name) {
                    match self.imports.get(&local_name) {
                        None => {
                            let message = format!(
                                "the function {} is not defined or imported in this module",
                                &local_name
                            );
                            self.has_errors = true;
                            self.diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("reference to undefined function")
                                .with_primary_label(span, message)
                                .emit();
                        }
                        Some(imported) => match self.app.get_function_deprecation(&imported) {
                            None => (),
                            Some(Deprecation::Module { span: dspan, flag }) => {
                                let note = format!("this module will be deprecated {}", &flag);
                                self.diagnostics
                                    .diagnostic(Severity::Warning)
                                    .with_message("use of deprecated module")
                                    .with_primary_label(span, note)
                                    .with_secondary_label(dspan, "deprecated here")
                                    .emit();
                            }
                            Some(Deprecation::Function {
                                span: dspan, flag, ..
                            }) => {
                                let note = format!("this function will be deprecated {}", &flag);
                                self.diagnostics
                                    .diagnostic(Severity::Warning)
                                    .with_message("use of deprecated function")
                                    .with_primary_label(span, note)
                                    .with_secondary_label(dspan, "deprecated here")
                                    .emit();
                            }
                            // These deprecation types have all been converted to Deprecation::Function
                            Some(Deprecation::FunctionAnyArity { .. }) => unreachable!(),
                        },
                    }
                }

                if name.arity > arity {
                    let message = format!(
                        "{} requires {} arguments, but only {} were provided",
                        name, name.arity, arity
                    );
                    self.has_errors = true;
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("missing arguments")
                        .with_primary_label(span, message)
                        .emit();
                } else if name.arity < arity {
                    let message = format!(
                        "{} only takes {} arguments, but {} were provided",
                        name, name.arity, arity
                    );
                    self.has_errors = true;
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("too many arguments")
                        .with_primary_label(span, message)
                        .emit();
                }
                ControlFlow::Continue(())
            }
            Expr::FunctionVar(FunctionVar::Unresolved(name)) => {
                if let Some(Name::Atom(m)) = name.module {
                    match self.app.get_module_deprecation(&m.name) {
                        Some(Deprecation::Module { span: dspan, flag }) => {
                            let note = format!("this module will be deprecated {}", &flag);
                            self.diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("use of deprecated module")
                                .with_primary_label(span, note)
                                .with_secondary_label(dspan, "deprecated here")
                                .emit();
                        }
                        _ => (),
                    }
                }
                if name.module.is_none() {
                    if let Name::Atom(a) = name.function {
                        let name = FunctionName::new_local(a.name, arity);
                        if !self.locals.contains(&name) {
                            match self.imports.get(&name) {
                                None => {
                                    let message = format!(
                                        "the function {} is not defined or imported in this module",
                                        &name
                                    );
                                    self.has_errors = true;
                                    self.diagnostics
                                        .diagnostic(Severity::Error)
                                        .with_message("reference to undefined function")
                                        .with_primary_label(span, message)
                                        .emit();
                                }
                                Some(imported) => {
                                    match self.app.get_function_deprecation(&imported) {
                                        None => (),
                                        Some(Deprecation::Module { span: dspan, flag }) => {
                                            let note =
                                                format!("this module will be deprecated {}", &flag);
                                            self.diagnostics
                                                .diagnostic(Severity::Warning)
                                                .with_message("use of deprecated module")
                                                .with_primary_label(span, note)
                                                .with_secondary_label(dspan, "deprecated here")
                                                .emit();
                                        }
                                        Some(Deprecation::Function {
                                            span: dspan, flag, ..
                                        }) => {
                                            let note = format!(
                                                "this function will be deprecated {}",
                                                &flag
                                            );
                                            self.diagnostics
                                                .diagnostic(Severity::Warning)
                                                .with_message("use of deprecated function")
                                                .with_primary_label(span, note)
                                                .with_secondary_label(dspan, "deprecated here")
                                                .emit();
                                        }
                                        // These deprecation types have all been converted to Deprecation::Function
                                        Some(Deprecation::FunctionAnyArity { .. }) => {
                                            unreachable!()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                match name.arity {
                    Arity::Int(i) if i > arity => {
                        let message = format!(
                            "this call requires {} arguments, but only {} were provided",
                            i, arity
                        );
                        self.has_errors = true;
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("missing arguments")
                            .with_primary_label(span, message)
                            .emit();
                    }
                    Arity::Int(i) if i < arity => {
                        let message = format!(
                            "this call should only have {} arguments, but {} were provided",
                            i, arity
                        );
                        self.has_errors = true;
                        self.diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("too many arguments")
                            .with_primary_label(span, message)
                            .emit();
                    }
                    _ => (),
                }
                ControlFlow::Continue(())
            }
            Expr::Literal(Literal::Atom(id)) => {
                let name = FunctionName::new_local(id.name, arity);
                if !self.locals.contains(&name) {
                    match self.imports.get(&name) {
                        None => {
                            let message =
                                format!("{} is not defined or imported in this module", &name);
                            self.has_errors = true;
                            self.diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("reference to undefined function")
                                .with_primary_label(span, message)
                                .emit();
                        }
                        Some(imported) => match self.app.get_function_deprecation(&imported) {
                            None => (),
                            Some(Deprecation::Module { span: dspan, flag }) => {
                                let note = format!("this module will be deprecated {}", &flag);
                                self.diagnostics
                                    .diagnostic(Severity::Warning)
                                    .with_message("use of deprecated module")
                                    .with_primary_label(span, note)
                                    .with_secondary_label(dspan, "deprecated here")
                                    .emit();
                            }
                            Some(Deprecation::Function {
                                span: dspan, flag, ..
                            }) => {
                                let note = format!("this function will be deprecated {}", &flag);
                                self.diagnostics
                                    .diagnostic(Severity::Warning)
                                    .with_message("use of deprecated function")
                                    .with_primary_label(span, note)
                                    .with_secondary_label(dspan, "deprecated here")
                                    .emit();
                            }
                            // These deprecation types have all been converted to Deprecation::Function
                            Some(Deprecation::FunctionAnyArity { .. }) => unreachable!(),
                        },
                    }
                }
                ControlFlow::Continue(())
            }
            _ => ControlFlow::Continue(()),
        }
    }
}
