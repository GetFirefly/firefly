use firefly_intern::Symbol;
use firefly_syntax_base::{bifs, CompileOptions, Deprecation, FunctionName, Signature};
use firefly_util::diagnostics::{DiagnosticsHandler, Severity};

use crate::ast::*;

use super::*;

pub fn analyze_attribute(diagnostics: &DiagnosticsHandler, module: &mut Module, attr: Attribute) {
    match attr {
        Attribute::Vsn(span, vsn) => {
            if module.vsn.is_none() {
                let vsn_span = vsn.span();
                let vsn_lit: Result<ast::Literal, _> = vsn.try_into();
                if vsn_lit.is_err() {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid -vsn attribute value")
                        .with_primary_label(span, "expected a literal value")
                        .with_secondary_label(vsn_span, "this expression is not a valid literal")
                        .emit();
                } else {
                    module.vsn = Some(vsn_lit.unwrap());
                }
                return;
            }
            let module_vsn_span = module.vsn.as_ref().map(|v| v.span()).unwrap();

            diagnostics
                .diagnostic(Severity::Error)
                .with_message("attribute is already defined")
                .with_primary_label(module_vsn_span, "first defined here")
                .with_secondary_label(span, "redefinition occurs here")
                .emit();
        }
        Attribute::Author(span, author) => {
            if module.author.is_none() {
                let author_span = author.span();
                let author_lit: Result<ast::Literal, _> = author.try_into();
                if author_lit.is_err() {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid -author attribute value")
                        .with_primary_label(span, "expected a literal value")
                        .with_secondary_label(author_span, "this expression is not a valid literal")
                        .emit();
                } else {
                    module.author = Some(author_lit.unwrap());
                }
                return;
            }
            let module_author_span = module.author.as_ref().map(|v| v.span()).unwrap();
            diagnostics
                .diagnostic(Severity::Error)
                .with_message("attribute is already defined")
                .with_primary_label(module_author_span, "first defined here")
                .with_secondary_label(span, "redefinition occurs here")
                .emit();
        }
        Attribute::OnLoad(span, fname) => {
            if module.on_load.is_none() {
                module.on_load = Some(Span::new(span, fname.to_local()));
                return;
            }
            let module_onload_span = module.on_load.as_ref().map(|v| v.span()).unwrap();
            diagnostics
                .diagnostic(Severity::Error)
                .with_message("on_load can only be defined once")
                .with_primary_label(module_onload_span, "first defined here")
                .with_secondary_label(span, "redefinition occurs here")
                .emit();
        }
        Attribute::Import(span, from_module, mut imports) => {
            for local_import in imports.drain(..) {
                let import = local_import.resolve(from_module.name);
                match module.imports.get(&local_import) {
                    None => {
                        let sig = match bifs::get(&import) {
                            Some(sig) => sig.clone(),
                            None => {
                                // Generate a default signature
                                Signature::generate(&import)
                            }
                        };
                        module.imports.insert(*local_import, Span::new(span, sig));
                    }
                    Some(ref spanned) => {
                        let prev_span = spanned.span();
                        diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("unused import")
                            .with_primary_label(
                                span,
                                "this import is a duplicate of a previous import",
                            )
                            .with_secondary_label(prev_span, "function was first imported here")
                            .emit();
                    }
                }
            }
        }
        Attribute::Export(span, mut exports) => {
            for export in exports.drain(..) {
                let local_export = Span::new(export.span(), export.to_local());
                match module.exports.get(&local_export) {
                    None => {
                        module.exports.insert(local_export);
                    }
                    Some(ref spanned) => {
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("already exported")
                            .with_primary_label(span, "duplicate export occurs here")
                            .with_secondary_label(
                                spanned.span(),
                                "function was first exported here",
                            )
                            .emit();
                    }
                }
            }
        }
        Attribute::Nifs(span, mut nifs) => {
            for nif in nifs.drain(..) {
                let local_export = Span::new(nif.span(), nif.to_local());
                match module.nifs.get(&local_export) {
                    None => {
                        module.nifs.insert(local_export);
                    }
                    Some(ref spanned) => {
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("duplicate -nif declaration")
                            .with_primary_label(span, "duplicate declaration occurs here")
                            .with_secondary_label(spanned.span(), "originally declared here")
                            .emit();
                    }
                }
            }
        }
        Attribute::Removed(span, mut removed) => {
            for (name, description) in removed.drain(..) {
                let local_name = name.to_local();
                if let Some((prev_span, _)) = module.removed.get(&local_name) {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("already marked as removed")
                        .with_primary_label(span, "duplicate entry occurs here")
                        .with_secondary_label(*prev_span, "function was marked here")
                        .emit();
                } else {
                    module.removed.insert(local_name, (span, description));
                }
            }
        }
        Attribute::Type(ty) => {
            let arity = ty.params.len();
            let type_name = FunctionName::new_local(ty.name.name, arity.try_into().unwrap());
            match module.types.get(&type_name) {
                None => {
                    module.types.insert(type_name, ty);
                }
                Some(TypeDef { span, .. }) => {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("type is already defined")
                        .with_primary_label(ty.span, "redefinition occurs here")
                        .with_secondary_label(*span, "type was first defined here")
                        .emit();
                }
            }
        }
        Attribute::ExportType(span, mut exports) => {
            for export in exports.drain(..) {
                let local_export = Span::new(export.span(), export.to_local());
                match module.exported_types.get(&local_export) {
                    None => {
                        module.exported_types.insert(local_export);
                    }
                    Some(ref spanned) => {
                        let prev_span = spanned.span();
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("type already exported")
                            .with_primary_label(span, "duplicate export occurs here")
                            .with_secondary_label(prev_span, "type was first exported here")
                            .emit();
                    }
                }
            }
        }
        Attribute::Behaviour(span, b_module) => match module.behaviours.get(&b_module) {
            None => {
                module.behaviours.insert(b_module);
            }
            Some(prev) => {
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("duplicate behavior declaration")
                    .with_primary_label(span, "duplicate declaration occurs here")
                    .with_secondary_label(prev.span, "first declaration occurs here")
                    .emit();
            }
        },
        Attribute::Callback(callback) => {
            let first_sig = callback.sigs.first().unwrap();
            let arity = first_sig.params.len();

            // Verify that all clauses match
            if callback.sigs.len() > 1 {
                for TypeSig {
                    span, ref params, ..
                } in &callback.sigs[1..]
                {
                    if params.len() != arity {
                        let message = format!("expected arity of {}", arity);
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("mismatched arity")
                            .with_primary_label(*span, message)
                            .with_secondary_label(
                                first_sig.span,
                                "expected arity was derived from this clause",
                            )
                            .emit();
                    }
                }
            }
            // Check for redefinition
            let cb_name = FunctionName::new(
                module.name(),
                callback.function.name,
                arity.try_into().unwrap(),
            );
            let local_cb_name = cb_name.to_local();
            match module.callbacks.get(&local_cb_name) {
                None => {
                    module.callbacks.insert(local_cb_name, callback);
                    return;
                }
                Some(ref prev_cb) => {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("cannot redefined callback")
                        .with_primary_label(callback.span, "redefinition occurs here")
                        .with_secondary_label(prev_cb.span, "callback first defined here")
                        .emit();
                }
            }
        }
        Attribute::Spec(typespec) => {
            let first_sig = typespec.sigs.first().unwrap();
            let arity = first_sig.params.len();

            // Verify that all clauses match
            if typespec.sigs.len() > 1 {
                for TypeSig {
                    span, ref params, ..
                } in &typespec.sigs[1..]
                {
                    if params.len() != arity {
                        let message = format!("expected arity of {}", arity);
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message("mismatched arity")
                            .with_primary_label(*span, message)
                            .with_secondary_label(
                                first_sig.span,
                                "expected arity was derived from this clause",
                            )
                            .emit();
                    }
                }
            }

            // Check for redefinition
            let spec_name = FunctionName::new(
                module.name(),
                typespec.function.name,
                arity.try_into().unwrap(),
            );
            match module.specs.get(&spec_name) {
                None => {
                    module.specs.insert(spec_name, typespec);
                }
                Some(TypeSpec { span, .. }) => {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("spec already defined")
                        .with_primary_label(typespec.span, "redefinition occurs here")
                        .with_secondary_label(*span, "spec first defined here")
                        .emit();
                }
            }
        }
        Attribute::Compile(_, compile) => match module.compile {
            None => match compile_opts_from_expr(module.name, &compile, diagnostics) {
                Ok(opts) => module.compile = Some(opts),
                Err(opts) => module.compile = Some(opts),
            },
            Some(ref mut opts) => {
                let _ = merge_compile_opts_from_expr(opts, module.name, &compile, diagnostics);
            }
        },
        Attribute::Deprecation(mut deprecations) => {
            for deprecation in deprecations.drain(..) {
                match deprecation {
                    Deprecation::Module { span, .. } => {
                        match module.deprecation {
                            None => {
                                module.deprecation = Some(deprecation);
                            }
                            Some(ref prev_dep) => {
                                diagnostics.diagnostic(Severity::Warning)
                                .with_message("redundant deprecation")
                                .with_primary_label(span, "this module is already deprecated by a previous declaration")
                                .with_secondary_label(prev_dep.span(), "deprecation first declared here")
                                .emit();
                            }
                        }
                    }
                    Deprecation::Function { span, .. } => {
                        if let Some(ref mod_dep) = module.deprecation.as_ref() {
                            diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("redundant deprecation")
                                .with_primary_label(
                                    span,
                                    "module is deprecated, so deprecating functions is redundant",
                                )
                                .with_secondary_label(
                                    mod_dep.span(),
                                    "module deprecation occurs here",
                                )
                                .emit();
                            return;
                        }

                        match module.deprecations.get(&deprecation) {
                            None => {
                                module.deprecations.insert(deprecation);
                            }
                            Some(ref prev_dep) => {
                                diagnostics.diagnostic(Severity::Warning)
                                    .with_message("redundant deprecation")
                                    .with_primary_label(span, "this function is already deprecated by a previous declaration")
                                    .with_secondary_label(prev_dep.span(), "deprecation first declared here")
                                    .emit();
                            }
                        }
                    }
                    Deprecation::FunctionAnyArity { span, .. } => {
                        if let Some(ref mod_dep) = module.deprecation.as_ref() {
                            diagnostics
                                .diagnostic(Severity::Warning)
                                .with_message("redundant deprecation")
                                .with_primary_label(
                                    span,
                                    "module is deprecated, so deprecating functions is redundant",
                                )
                                .with_secondary_label(
                                    mod_dep.span(),
                                    "module deprecation occurs here",
                                )
                                .emit();
                            return;
                        }

                        match module.deprecations.get(&deprecation) {
                            None => {
                                module.deprecations.insert(deprecation);
                            }
                            Some(ref prev_dep) => {
                                diagnostics
                                    .diagnostic(Severity::Warning)
                                    .with_message("conflicting deprecation")
                                    .with_primary_label(
                                        span,
                                        "this deprecation is a duplicate of a previous declaration",
                                    )
                                    .with_secondary_label(prev_dep.span(), "first declared here")
                                    .emit();
                            }
                        }
                    }
                }
            }
        }
        Attribute::Custom(attr) => {
            match attr.name.name.as_str().get() {
                "module" => {
                    diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("multiple module declarations")
                        .with_primary_label(attr.span, "invalid declaration occurs here")
                        .with_secondary_label(module.name.span, "module first declared here")
                        .emit();
                    return;
                }
                "optional_callbacks" => {
                    return;
                }
                // Drop dialyzer attributes as they are unused
                "dialyzer" => {
                    return;
                }
                _ => (),
            }
            let attr_value: Result<ast::Literal, _> = attr.value.try_into();
            if attr_value.is_err() {
                diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("invalid attribute value")
                    .with_primary_label(attr.span, "attribute values must be literals")
                    .with_note("this attribute will be ignored")
                    .emit();
                return;
            }
            match module.attributes.get(&attr.name) {
                None => {
                    module
                        .attributes
                        .insert(attr.name.clone(), attr_value.unwrap());
                }
                Some(ref prev_attr) => {
                    diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("redefined attribute")
                        .with_primary_label(attr.span, "redefinition occurs here")
                        .with_secondary_label(prev_attr.span(), "previously defined here")
                        .emit();
                    module.attributes.insert(attr.name, attr_value.unwrap());
                }
            }
        }
    }
}

fn compile_opts_from_expr(
    module: Ident,
    expr: &Expr,
    diagnostics: &DiagnosticsHandler,
) -> Result<CompileOptions, CompileOptions> {
    let mut opts = CompileOptions::default();
    match merge_compile_opts_from_expr(&mut opts, module, expr, diagnostics) {
        Ok(_) => Ok(opts),
        Err(_) => Err(opts),
    }
}

fn merge_compile_opts_from_expr(
    options: &mut CompileOptions,
    module: Ident,
    expr: &Expr,
    diagnostics: &DiagnosticsHandler,
) -> Result<(), ()> {
    set_compile_option(options, module, expr, diagnostics)
}

fn set_compile_option(
    options: &mut CompileOptions,
    module: Ident,
    expr: &Expr,
    diagnostics: &DiagnosticsHandler,
) -> Result<(), ()> {
    match expr {
        // e.g. -compile(export_all).
        &Expr::Literal(Literal::Atom(ref option_name)) => {
            match option_name.as_str().get() {
                "no_native" => (), // Disables hipe compilation, not relevant for us
                "inline" => options.inline = true,

                "export_all" => options.export_all = true,

                "no_auto_import" => options.no_auto_import = true,

                "report_errors" => options.report_errors = true,
                "report_warnings" => options.report_errors = true,
                "verbose" => options.verbose = true,

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
                        options.inline_functions.insert(Span::new(
                            option_name.span,
                            FunctionName::new(Symbol::intern(m), Symbol::intern(f), *a),
                        ));
                    }
                }

                // Warning toggles
                "warn_export_all" => options.warn_export_all = true,
                "nowarn_export_all" => options.warn_export_all = false,

                "warn_shadow_vars" => options.warn_shadow_vars = true,
                "nowarn_shadow_vars" => options.warn_shadow_vars = false,

                "warn_unused_function" => options.warn_unused_function = true,
                "nowarn_unused_function" => options.warn_unused_function = false,

                "warn_unused_import" => options.warn_unused_import = true,
                "nowarn_unused_import" => options.warn_unused_import = false,

                "warn_unused_type" => options.warn_unused_type = true,
                "nowarn_unused_type" => options.warn_unused_type = false,

                "warn_export_vars" => options.warn_export_vars = true,
                "nowarn_export_vars" => options.warn_export_vars = false,

                "warn_unused_vars" => options.warn_unused_var = true,
                "nowarn_unused_vars" => options.warn_unused_var = false,

                "warn_bif_clash" => options.warn_bif_clash = true,
                "nowarn_bif_clash" => options.warn_bif_clash = false,

                "warn_unused_record" => options.warn_unused_record = true,
                "nowarn_unused_record" => options.warn_unused_record = false,

                "warn_deprecated_function" => options.warn_deprecated_functions = true,
                "nowarn_deprecated_function" => options.warn_deprecated_functions = false,

                "warn_deprecated_type" => options.warn_deprecated_type = true,
                "nowarn_deprecated_type" => options.warn_deprecated_type = false,

                "warn_obsolete_guard" => options.warn_obsolete_guard = true,
                "nowarn_obsolete_guard" => options.warn_obsolete_guard = false,

                "warn_untyped_record" => options.warn_untyped_record = true,
                "nowarn_untyped_record" => options.warn_untyped_record = false,

                "warn_missing_spec" => options.warn_missing_spec = true,
                "nowarn_missing_spec" => options.warn_missing_spec = false,

                "warn_missing_spec_all" => options.warn_missing_spec_all = true,
                "nowarn_missing_spec_all" => options.warn_missing_spec_all = false,

                "warn_removed" => options.warn_removed = true,
                "nowarn_removed" => options.warn_removed = false,

                "warn_nif_inline" => options.warn_nif_inline = true,
                "nowarn_nif_inline" => options.warn_nif_inline = false,

                _name => {
                    diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("invalid compile option")
                        .with_primary_label(
                            option_name.span,
                            "this option is either unsupported or unrecognized",
                        )
                        .emit();
                    return Err(());
                }
            }
        }
        // e.g. -compile([export_all, nowarn_unused_function]).
        &Expr::Cons(Cons {
            ref head, ref tail, ..
        }) => compiler_opts_from_list(options, module, to_list(head, tail), diagnostics),
        // e.g. -compile({nowarn_unused_function, [some_fun/0]}).
        &Expr::Tuple(Tuple { ref elements, .. }) if elements.len() == 2 => {
            if let &Expr::Literal(Literal::Atom(ref option_name)) = &elements[0] {
                let list = to_list_simple(&elements[1]);
                match option_name.as_str().get() {
                    "no_auto_import" => no_auto_imports(options, module, &list, diagnostics),
                    "nowarn_unused_function" => {
                        no_warn_unused_functions(options, module, &list, diagnostics)
                    }
                    "nowarn_deprecated_function" => {
                        no_warn_deprecated_functions(options, module, &list, diagnostics)
                    }
                    "inline" => inline_functions(options, module, &list, diagnostics),
                    // Ignored
                    "hipe" => {}
                    _name => {
                        diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("invalid compile option")
                            .with_primary_label(
                                option_name.span,
                                "this option is either unsupported or unrecognized",
                            )
                            .emit();
                        return Err(());
                    }
                }
            }
        }
        term => {
            let term_span = term.span();
            diagnostics
                .diagnostic(Severity::Warning)
                .with_message("invalid compile option")
                .with_primary_label(
                    term_span,
                    "unexpected expression: expected atom, list, or tuple",
                )
                .emit();
            return Err(());
        }
    }

    Ok(())
}

fn compiler_opts_from_list(
    options: &mut CompileOptions,
    module: Ident,
    list: Vec<Expr>,
    diagnostics: &DiagnosticsHandler,
) {
    for option in list.iter() {
        let _ = set_compile_option(options, module, option, diagnostics);
    }
}

fn no_auto_imports(
    options: &mut CompileOptions,
    module: Ident,
    imports: &[Expr],
    diagnostics: &DiagnosticsHandler,
) {
    for import in imports {
        match import {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                options.no_auto_imports.insert(name.resolve(module.name));
            }
            Expr::Tuple(tup) if tup.elements.len() == 2 => {
                match (&tup.elements[0], &tup.elements[1]) {
                    (
                        Expr::Literal(Literal::Atom(name)),
                        Expr::Literal(Literal::Integer(_, arity)),
                    ) => {
                        let name = FunctionName::new(module.name, name.name, arity.to_arity());
                        options.no_auto_imports.insert(name);
                        continue;
                    }
                    _ => (),
                }
            }
            other => {
                let other_span = other.span();
                diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("invalid compile option")
                    .with_primary_label(
                        other_span,
                        "expected function name/arity term for no_auto_imports",
                    )
                    .emit();
            }
        }
    }
}

fn no_warn_unused_functions(
    options: &mut CompileOptions,
    _module: Ident,
    funs: &[Expr],
    diagnostics: &DiagnosticsHandler,
) {
    for fun in funs {
        match fun {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                options.no_warn_unused_functions.insert(*name);
            }
            other => {
                let other_span = other.span();
                diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("invalid compile option")
                    .with_primary_label(
                        other_span,
                        "expected function name/arity term for no_warn_unused_functions",
                    )
                    .emit();
            }
        }
    }
}

fn no_warn_deprecated_functions(
    options: &mut CompileOptions,
    _module: Ident,
    funs: &[Expr],
    diagnostics: &DiagnosticsHandler,
) {
    use firefly_number::Int;

    for fun in funs {
        match fun {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                options.no_warn_deprecated_functions.insert(*name);
            }
            Expr::Literal(Literal::Tuple(span, ref elements)) if elements.len() == 3 => {
                match elements.as_slice() {
                    [Literal::Atom(m), Literal::Atom(f), Literal::Integer(_, Int::Small(a))] => {
                        options.no_warn_deprecated_functions.insert(Span::new(
                            *span,
                            FunctionName::new(m.name, f.name, (*a).try_into().unwrap()),
                        ));
                    }
                    _ => {
                        diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("invalid compile option")
                            .with_primary_label(
                                *span,
                                "expected name/arity or mfa tuple for no_warn_deprecated_function",
                            )
                            .emit();
                    }
                }
            }
            Expr::Tuple(Tuple { span, ref elements }) if elements.len() == 3 => {
                match elements.as_slice() {
                    [Expr::Literal(Literal::Atom(m)), Expr::Literal(Literal::Atom(f)), Expr::Literal(Literal::Integer(_, Int::Small(a)))] =>
                    {
                        options.no_warn_deprecated_functions.insert(Span::new(
                            *span,
                            FunctionName::new(m.name, f.name, (*a).try_into().unwrap()),
                        ));
                    }
                    _ => {
                        diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("invalid compile option")
                            .with_primary_label(
                                *span,
                                "expected name/arity or mfa tuple for no_warn_deprecated_function",
                            )
                            .emit();
                    }
                }
            }
            other => {
                let other_span = other.span();
                diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("invalid compile option")
                    .with_primary_label(
                        other_span,
                        "expected name/arity or mfa tuple for no_warn_deprecated_function",
                    )
                    .emit();
            }
        }
    }
}

fn inline_functions(
    options: &mut CompileOptions,
    module: Ident,
    funs: &[Expr],
    diagnostics: &DiagnosticsHandler,
) {
    for fun in funs {
        match fun {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                let name = Span::new(name.span(), name.resolve(module.name));
                options.inline_functions.insert(name);
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
                            FunctionName::new(module.name, name.name, arity.to_arity()),
                        );
                        options.inline_functions.insert(name);
                        continue;
                    }
                    _ => (),
                }
            }
            _ => (),
        }

        let fun_span = fun.span();
        diagnostics
            .diagnostic(Severity::Warning)
            .with_message("invalid compile option")
            .with_primary_label(fun_span, "expected function name/arity term for inline")
            .emit();
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
