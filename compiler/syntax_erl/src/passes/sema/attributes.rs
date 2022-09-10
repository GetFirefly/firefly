use firefly_diagnostics::Reporter;
use firefly_intern::Symbol;
use firefly_syntax_base::{bifs, CompileOptions, Deprecation, FunctionName, Signature};

use crate::ast::*;

use super::*;

pub fn analyze_attribute(reporter: &Reporter, module: &mut Module, attr: Attribute) {
    match attr {
        Attribute::Vsn(span, vsn) => {
            if module.vsn.is_none() {
                let vsn_span = vsn.span();
                let vsn_lit: Result<ast::Literal, _> = vsn.try_into();
                if vsn_lit.is_err() {
                    reporter.show_error(
                        "invalid -vsn attribute value",
                        &[
                            (span, "expected a literal value"),
                            (vsn_span, "this expression is not a valid literal"),
                        ],
                    );
                } else {
                    module.vsn = Some(vsn_lit.unwrap());
                }
                return;
            }
            let module_vsn_span = module.vsn.as_ref().map(|v| v.span()).unwrap();
            reporter.show_error(
                "attribute is already defined",
                &[
                    (span, "redefinition occurs here"),
                    (module_vsn_span, "first defined here"),
                ],
            );
        }
        Attribute::Author(span, author) => {
            if module.author.is_none() {
                let author_span = author.span();
                let author_lit: Result<ast::Literal, _> = author.try_into();
                if author_lit.is_err() {
                    reporter.show_error(
                        "invalid -author attribute value",
                        &[
                            (span, "expected a literal value"),
                            (author_span, "this expression is not a valid literal"),
                        ],
                    );
                } else {
                    module.author = Some(author_lit.unwrap());
                }
                return;
            }
            let module_author_span = module.author.as_ref().map(|v| v.span()).unwrap();
            reporter.show_error(
                "attribute is already defined",
                &[
                    (span, "redefinition occurs here"),
                    (module_author_span, "first defined here"),
                ],
            );
        }
        Attribute::OnLoad(span, fname) => {
            if module.on_load.is_none() {
                module.on_load = Some(Span::new(span, fname.to_local()));
                return;
            }
            let module_onload_span = module.on_load.as_ref().map(|v| v.span()).unwrap();
            reporter.show_error(
                "on_load can only be defined once",
                &[
                    (span, "redefinition occurs here"),
                    (module_onload_span, "first defined here"),
                ],
            );
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
                        reporter.show_warning(
                            "unused import",
                            &[
                                (span, "this import is a duplicate of a previous import"),
                                (prev_span, "function was first imported here"),
                            ],
                        );
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
                        reporter.show_error(
                            "already exported",
                            &[
                                (span, "duplicate export occurs here"),
                                (spanned.span(), "function was first exported here"),
                            ],
                        );
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
                        reporter.show_error(
                            "duplicate -nif declaration",
                            &[
                                (span, "duplicate declaration occurs here"),
                                (spanned.span(), "originally declared here"),
                            ],
                        );
                    }
                }
            }
        }
        Attribute::Removed(span, mut removed) => {
            for (name, description) in removed.drain(..) {
                let local_name = name.to_local();
                if let Some((prev_span, _)) = module.removed.get(&local_name) {
                    reporter.show_error(
                        "already marked as removed",
                        &[
                            (span, "duplicate entry occurs here"),
                            (*prev_span, "function was marked here"),
                        ],
                    );
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
                    reporter.show_error(
                        "type is already defined",
                        &[
                            (ty.span, "redefinition occurs here"),
                            (*span, "type was first defined here"),
                        ],
                    );
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
                        reporter.show_warning(
                            "type already exported",
                            &[
                                (span, "duplicate export occurs here"),
                                (prev_span, "type was first exported here"),
                            ],
                        );
                    }
                }
            }
        }
        Attribute::Behaviour(span, b_module) => match module.behaviours.get(&b_module) {
            None => {
                module.behaviours.insert(b_module);
            }
            Some(prev) => {
                reporter.show_warning(
                    "duplicate behavior declaration",
                    &[
                        (span, "duplicate declaration occurs here"),
                        (prev.span, "first declaration occurs here"),
                    ],
                );
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
                        reporter.show_error(
                            "mismatched arity",
                            &[
                                (*span, message.as_str()),
                                (
                                    first_sig.span,
                                    "expected arity was derived from this clause",
                                ),
                            ],
                        );
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
                    reporter.show_error(
                        "cannot redefined callback",
                        &[
                            (callback.span, "redefinition occurs here"),
                            (prev_cb.span, "callback first defined here"),
                        ],
                    );
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
                        reporter.show_error(
                            "mismatched arity",
                            &[
                                (*span, message.as_str()),
                                (
                                    first_sig.span,
                                    "expected arity was derived from this clause",
                                ),
                            ],
                        );
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
                    reporter.show_error(
                        "spec already defined",
                        &[
                            (typespec.span, "redefinition occurs here"),
                            (*span, "spec first defined here"),
                        ],
                    );
                }
            }
        }
        Attribute::Compile(_, compile) => match module.compile {
            None => match compile_opts_from_expr(module.name, &compile, reporter) {
                Ok(opts) => module.compile = Some(opts),
                Err(opts) => module.compile = Some(opts),
            },
            Some(ref mut opts) => {
                let _ = merge_compile_opts_from_expr(opts, module.name, &compile, reporter);
            }
        },
        Attribute::Deprecation(mut deprecations) => {
            for deprecation in deprecations.drain(..) {
                match deprecation {
                    Deprecation::Module { span, .. } => match module.deprecation {
                        None => {
                            module.deprecation = Some(deprecation);
                        }
                        Some(ref prev_dep) => {
                            reporter.show_warning("redundant deprecation", &[(span, "this module is already deprecated by a previous declaration"), (prev_dep.span(), "deprecation first declared here")]);
                        }
                    },
                    Deprecation::Function { span, .. } => {
                        if let Some(ref mod_dep) = module.deprecation.as_ref() {
                            reporter.show_warning("redundant deprecation", &[(span, "module is deprecated, so deprecating functions is redundant"), (mod_dep.span(), "module deprecation occurs here")]);
                            return;
                        }

                        match module.deprecations.get(&deprecation) {
                            None => {
                                module.deprecations.insert(deprecation);
                            }
                            Some(ref prev_dep) => {
                                reporter.show_warning("redundant deprecation", &[(span, "this function is already deprecated by a previous declaration"), (prev_dep.span(), "deprecation first declared here")]);
                            }
                        }
                    }
                    Deprecation::FunctionAnyArity { span, .. } => {
                        if let Some(ref mod_dep) = module.deprecation.as_ref() {
                            reporter.show_warning("redundant deprecation", &[(span, "module is deprecated, so deprecating functions is redundant"), (mod_dep.span(), "module deprecation occurs here")]);
                            return;
                        }

                        match module.deprecations.get(&deprecation) {
                            None => {
                                module.deprecations.insert(deprecation);
                            }
                            Some(ref prev_dep) => {
                                reporter.show_warning("conflicting deprecation", &[(span, "this deprecation is a duplicate of a previous declaration"), (prev_dep.span(), "first declared here")]);
                            }
                        }
                    }
                }
            }
        }
        Attribute::Custom(attr) => {
            match attr.name.name.as_str().get() {
                "module" => {
                    reporter.show_error(
                        "multiple module declarations",
                        &[
                            (attr.span, "invalid declaration occurs here"),
                            (module.name.span, "module first declared here"),
                        ],
                    );
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
                reporter.show_warning(
                    "invalid attribute value",
                    &[
                        (attr.span, "attribute values must be literals"),
                        (attr.span, "this attribute will be ignored"),
                    ],
                );
                return;
            }
            match module.attributes.get(&attr.name) {
                None => {
                    module
                        .attributes
                        .insert(attr.name.clone(), attr_value.unwrap());
                }
                Some(ref prev_attr) => {
                    reporter.show_warning(
                        "redefined attribute",
                        &[
                            (attr.span, "redefinition occurs here"),
                            (prev_attr.span(), "previously defined here"),
                        ],
                    );
                    module.attributes.insert(attr.name, attr_value.unwrap());
                }
            }
        }
    }
}

fn compile_opts_from_expr(
    module: Ident,
    expr: &Expr,
    reporter: &Reporter,
) -> Result<CompileOptions, CompileOptions> {
    let mut opts = CompileOptions::default();
    match merge_compile_opts_from_expr(&mut opts, module, expr, reporter) {
        Ok(_) => Ok(opts),
        Err(_) => Err(opts),
    }
}

fn merge_compile_opts_from_expr(
    options: &mut CompileOptions,
    module: Ident,
    expr: &Expr,
    reporter: &Reporter,
) -> Result<(), ()> {
    set_compile_option(options, module, expr, reporter)
}

fn set_compile_option(
    options: &mut CompileOptions,
    module: Ident,
    expr: &Expr,
    reporter: &Reporter,
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
                    reporter.diagnostic(
                        Diagnostic::warning()
                            .with_message("invalid compile option")
                            .with_labels(vec![Label::primary(
                                option_name.span.source_id(),
                                option_name.span,
                            )
                            .with_message("this option is either unsupported or unrecognized")]),
                    );
                    return Err(());
                }
            }
        }
        // e.g. -compile([export_all, nowarn_unused_function]).
        &Expr::Cons(Cons {
            ref head, ref tail, ..
        }) => compiler_opts_from_list(options, module, to_list(head, tail), reporter),
        // e.g. -compile({nowarn_unused_function, [some_fun/0]}).
        &Expr::Tuple(Tuple { ref elements, .. }) if elements.len() == 2 => {
            if let &Expr::Literal(Literal::Atom(ref option_name)) = &elements[0] {
                let list = to_list_simple(&elements[1]);
                match option_name.as_str().get() {
                    "no_auto_import" => no_auto_imports(options, module, &list, reporter),
                    "nowarn_unused_function" => {
                        no_warn_unused_functions(options, module, &list, reporter)
                    }
                    "nowarn_deprecated_function" => {
                        no_warn_deprecated_functions(options, module, &list, reporter)
                    }
                    "inline" => inline_functions(options, module, &list, reporter),
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

fn compiler_opts_from_list(
    options: &mut CompileOptions,
    module: Ident,
    list: Vec<Expr>,
    reporter: &Reporter,
) {
    for option in list.iter() {
        let _ = set_compile_option(options, module, option, reporter);
    }
}

fn no_auto_imports(
    options: &mut CompileOptions,
    module: Ident,
    imports: &[Expr],
    reporter: &Reporter,
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

fn no_warn_unused_functions(
    options: &mut CompileOptions,
    _module: Ident,
    funs: &[Expr],
    reporter: &Reporter,
) {
    for fun in funs {
        match fun {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                options.no_warn_unused_functions.insert(*name);
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

fn no_warn_deprecated_functions(
    options: &mut CompileOptions,
    _module: Ident,
    funs: &[Expr],
    reporter: &Reporter,
) {
    use firefly_number::Integer;

    for fun in funs {
        match fun {
            Expr::FunctionVar(FunctionVar::PartiallyResolved(name)) => {
                options.no_warn_deprecated_functions.insert(*name);
            }
            Expr::Literal(Literal::Tuple(span, ref elements)) if elements.len() == 3 => {
                match elements.as_slice() {
                    [Literal::Atom(m), Literal::Atom(f), Literal::Integer(_, Integer::Small(a))] => {
                        options.no_warn_deprecated_functions.insert(Span::new(
                            *span,
                            FunctionName::new(m.name, f.name, (*a).try_into().unwrap()),
                        ));
                    }
                    _ => reporter.diagnostic(
                        Diagnostic::warning()
                            .with_message("invalid compile option")
                            .with_labels(vec![Label::primary(span.source_id(), *span)
                                .with_message(
                                "expected name/arity or mfa tuple for no_warn_deprecated_function",
                            )]),
                    ),
                }
            }
            Expr::Tuple(Tuple { span, ref elements }) if elements.len() == 3 => {
                match elements.as_slice() {
                    [Expr::Literal(Literal::Atom(m)), Expr::Literal(Literal::Atom(f)), Expr::Literal(Literal::Integer(_, Integer::Small(a)))] =>
                    {
                        options.no_warn_deprecated_functions.insert(Span::new(
                            *span,
                            FunctionName::new(m.name, f.name, (*a).try_into().unwrap()),
                        ));
                    }
                    _ => reporter.diagnostic(
                        Diagnostic::warning()
                            .with_message("invalid compile option")
                            .with_labels(vec![Label::primary(span.source_id(), *span)
                                .with_message(
                                "expected name/arity or mfa tuple for no_warn_deprecated_function",
                            )]),
                    ),
                }
            }
            other => {
                let other_span = other.span();
                reporter.diagnostic(
                    Diagnostic::warning()
                        .with_message("invalid compile option")
                        .with_labels(vec![Label::primary(other_span.source_id(), other_span)
                            .with_message(
                                "expected name/arity or mfa tuple for no_warn_deprecated_function",
                            )]),
                );
            }
        }
    }
}

fn inline_functions(
    options: &mut CompileOptions,
    module: Ident,
    funs: &[Expr],
    reporter: &Reporter,
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
        reporter.diagnostic(
            Diagnostic::warning()
                .with_message("invalid compile option")
                .with_labels(vec![Label::primary(fun_span.source_id(), fun_span)
                    .with_message("expected function name/arity term for inline")]),
        );
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
