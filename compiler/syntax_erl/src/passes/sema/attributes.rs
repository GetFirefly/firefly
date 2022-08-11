use liblumen_syntax_ssa as syntax_ssa;

use crate::ast::*;

use super::*;

impl SemanticAnalysis {
    pub(super) fn analyze_attribute(&mut self, module: &mut Module, attr: Attribute) {
        match attr {
            Attribute::Vsn(span, vsn) => {
                if module.vsn.is_none() {
                    module.vsn = Some(vsn);
                    return;
                }
                let module_vsn_span = module.vsn.as_ref().map(|v| v.span()).unwrap();
                self.reporter.show_error(
                    "attribute is already defined",
                    &[
                        (span, "redefinition occurs here"),
                        (module_vsn_span, "first defined here"),
                    ],
                );
            }
            Attribute::Author(span, author) => {
                if module.author.is_none() {
                    module.author = Some(author);
                    return;
                }
                let module_author_span = module.author.as_ref().map(|v| v.span()).unwrap();
                self.reporter.show_error(
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
                self.reporter.show_error(
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
                            let sig = match liblumen_syntax_ssa::bifs::get(&import) {
                                Some(sig) => sig.clone(),
                                None => {
                                    // Generate a default signature
                                    liblumen_syntax_ssa::Signature::generate(&import)
                                }
                            };
                            module.imports.insert(*local_import, Span::new(span, sig));
                        }
                        Some(ref spanned) => {
                            let prev_span = spanned.span();
                            self.reporter.show_warning(
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
                            self.reporter.show_error(
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
                            self.reporter.show_error(
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
                        self.reporter.show_error(
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
                let type_name =
                    syntax_ssa::FunctionName::new_local(ty.name.name, arity.try_into().unwrap());
                match module.types.get(&type_name) {
                    None => {
                        module.types.insert(type_name, ty);
                    }
                    Some(TypeDef { span, .. }) => {
                        self.reporter.show_error(
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
                            self.reporter.show_warning(
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
                    self.reporter.show_warning(
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
                            self.reporter.show_error(
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
                let cb_name = syntax_ssa::FunctionName::new(
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
                        self.reporter.show_error(
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
                            self.reporter.show_error(
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
                let spec_name = syntax_ssa::FunctionName::new(
                    module.name(),
                    typespec.function.name,
                    arity.try_into().unwrap(),
                );
                match module.specs.get(&spec_name) {
                    None => {
                        module.specs.insert(spec_name, typespec);
                    }
                    Some(TypeSpec { span, .. }) => {
                        self.reporter.show_error(
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
                None => match CompileOptions::from_expr(module.name, &compile, &self.reporter) {
                    Ok(opts) => module.compile = Some(opts),
                    Err(opts) => module.compile = Some(opts),
                },
                Some(ref mut opts) => {
                    let _ = opts.merge_from_expr(module.name, &compile, &self.reporter);
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
                                self.reporter.show_warning("redundant deprecation", &[(span, "this module is already deprecated by a previous declaration"), (prev_dep.span(), "deprecation first declared here")]);
                            }
                        },
                        Deprecation::Function { span, .. } => {
                            if let Some(ref mod_dep) = module.deprecation.as_ref() {
                                self.reporter.show_warning("redundant deprecation", &[(span, "module is deprecated, so deprecating functions is redundant"), (mod_dep.span(), "module deprecation occurs here")]);
                                return;
                            }

                            match module.deprecations.get(&deprecation) {
                                None => {
                                    module.deprecations.insert(deprecation);
                                }
                                Some(ref prev_dep) => {
                                    self.reporter.show_warning("redundant deprecation", &[(span, "this function is already deprecated by a previous declaration"), (prev_dep.span(), "deprecation first declared here")]);
                                }
                            }
                        }
                    }
                }
            }
            Attribute::Custom(attr) => {
                match attr.name.name.as_str().get() {
                    "module" => {
                        self.reporter.show_error(
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
                match module.attributes.get(&attr.name) {
                    None => {
                        module.attributes.insert(attr.name.clone(), attr);
                    }
                    Some(ref prev_attr) => {
                        self.reporter.show_warning(
                            "redefined attribute",
                            &[
                                (attr.span, "redefinition occurs here"),
                                (prev_attr.span, "previously defined here"),
                            ],
                        );
                        module.attributes.insert(attr.name, attr);
                    }
                }
            }
        }
    }
}
