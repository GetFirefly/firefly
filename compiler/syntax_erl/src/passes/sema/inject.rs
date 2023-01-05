use firefly_diagnostics::*;
use firefly_intern::symbols;
use firefly_pass::Pass;
use firefly_syntax_base::*;

use crate::ast::{self, *};

/// Registers auto-imported BIFs in the given module
///
/// This pass takes into account the compiler options of the module when deciding what to import
pub struct AddAutoImports;
impl Pass for AddAutoImports {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        if let Some(compile) = module.compile.as_ref() {
            if compile.no_auto_import {
                return Ok(module);
            }

            let span = module.name.span;
            for sig in bifs::all().iter().map(|sig| Span::new(span, sig.clone())) {
                let local_name = sig.mfa().to_local();
                if !compile.no_auto_imports.contains(&local_name) {
                    module.imports.insert(local_name, sig);
                }
            }
        } else {
            let span = module.name.span;
            for sig in bifs::all().iter().map(|sig| Span::new(span, sig.clone())) {
                let local_name = sig.mfa().to_local();
                module.imports.insert(local_name, sig);
            }
        }

        Ok(module)
    }
}

/// Every module in Erlang has some functions implicitly defined for internal use:
///
/// * `module_info/0` (exported)
/// * `module_info/1` (exported)
/// * `record_info/2`
/// * `behaviour_info/1` (optional)
///
/// NOTE: We do not provide the `md5` module info key, as its definition in Erlang doesn't
/// mean anything for us, and producing our own has no known benefit at this time.
pub struct DefinePseudoLocals;
impl Pass for DefinePseudoLocals {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Build up list of attributes for module_info
        let mut attributes = ast_lit_nil!();
        if let Some(vsn) = module.vsn.take() {
            let span = vsn.span();
            let key = ast_lit_atom!(span, symbols::Vsn);
            let value = ast_lit_tuple_with_span!(span, key, vsn);

            attributes = ast_lit_cons!(value, attributes);
        }
        if let Some(author) = module.author.take() {
            let span = author.span();
            let key = ast_lit_atom!(span, symbols::Author);
            let value = ast_lit_tuple_with_span!(span, key, author);

            attributes = ast_lit_cons!(value, attributes);
        }
        for (name, value) in module.attributes.drain() {
            let span = name.span;
            let key = ast::Literal::Atom(name);
            let value = ast_lit_tuple_with_span!(span, key, value);

            attributes =
                ast::Literal::Cons(SourceSpan::UNKNOWN, Box::new(value), Box::new(attributes));
        }

        // Build up list of exports in {name, arity} form for module_info
        let exports = module.exports.iter().fold(ast_lit_nil!(), |tail, export| {
            let name = ast_lit_atom!(export.span(), export.function);
            let arity = ast_lit_int!(export.span(), export.arity.into());

            ast_lit_cons!(ast_lit_tuple_with_span!(export.span(), name, arity), tail)
        });

        // Build up list of functions in {name, arity} form for module_info
        let functions = module
            .functions
            .iter()
            .fold(ast_lit_nil!(), |tail, (name, f)| {
                let fname = ast_lit_atom!(f.name.span, name.function);
                let arity = ast_lit_int!(f.name.span, name.arity.into());

                ast_lit_cons!(ast_lit_tuple_with_span!(f.name.span, fname, arity), tail)
            });

        // Build up list of nifs in {name, arity} form for module_info
        let nifs = module.nifs.iter().fold(ast_lit_nil!(), |tail, nif| {
            let fname = ast_lit_atom!(nif.span(), nif.function);
            let arity = ast_lit_int!(nif.span(), nif.arity.into());

            ast_lit_cons!(ast_lit_tuple_with_span!(nif.span(), fname, arity), tail)
        });

        // Define module_info/0 which contains a proplist with keys: module, attributes, compile,
        // exports, md5 and native
        let mod_info_0_list = ast_lit_list!(
            ast_lit_tuple!(
                ast_lit_atom!(symbols::Module),
                ast_lit_atom!(module.name.name)
            ),
            ast_lit_tuple!(ast_lit_atom!(symbols::Attributes), attributes.clone()),
            ast_lit_tuple!(ast_lit_atom!(symbols::Compile), ast_lit_nil!()),
            ast_lit_tuple!(ast_lit_atom!(symbols::Exports), exports.clone()),
            ast_lit_tuple!(ast_lit_atom!(symbols::Native), ast_lit_atom!(symbols::True))
        );
        let mod_info_0 = Function {
            span: SourceSpan::UNKNOWN,
            name: ident!(module_info),
            arity: 0,
            clauses: vec![(
                Some(Name::Atom(ident!(module_info))),
                Clause {
                    span: SourceSpan::UNKNOWN,
                    patterns: vec![],
                    guards: vec![],
                    body: vec![Expr::Literal(mod_info_0_list)],
                    compiler_generated: true,
                },
            )],
            spec: None,
            is_nif: false,
            var_counter: 0,
            fun_counter: 0,
        };
        define_function(module, mod_info_0);

        // Define module_info/1 which contains accepts the following keys: module, attributes,
        // compile, exports, functions, nifs, md5 and native
        let mod_info_1 = Function {
            span: SourceSpan::UNKNOWN,
            name: ident!(module_info),
            arity: 1,
            clauses: vec![
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Module))],
                        guards: vec![],
                        body: vec![Expr::Literal(ast_lit_atom!(module.name.name))],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Attributes))],
                        guards: vec![],
                        body: vec![Expr::Literal(attributes)],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Compile))],
                        guards: vec![],
                        body: vec![Expr::Literal(ast_lit_nil!())],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Exports))],
                        guards: vec![],
                        body: vec![Expr::Literal(exports)],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Functions))],
                        guards: vec![],
                        body: vec![Expr::Literal(functions)],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Nifs))],
                        guards: vec![],
                        body: vec![Expr::Literal(nifs)],
                        compiler_generated: true,
                    },
                ),
                // This clause is to avoid causing crashes in code which asks for this value,
                // but we don't make guarantees about it being meaningful
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Md5))],
                        guards: vec![],
                        body: vec![Expr::Literal(ast_lit_nil!())],
                        compiler_generated: true,
                    },
                ),
                (
                    Some(Name::Atom(ident!(module_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![Expr::Literal(ast_lit_atom!(symbols::Native))],
                        guards: vec![],
                        body: vec![Expr::Literal(ast_lit_atom!(symbols::True))],
                        compiler_generated: true,
                    },
                ),
            ],
            spec: None,
            is_nif: false,
            var_counter: 0,
            fun_counter: 0,
        };
        define_function(module, mod_info_1);

        if !module.records.is_empty() {
            let mut clauses = Vec::with_capacity(module.records.len() * 2);
            for record in module.records.values() {
                let size = (record.fields.len() + 1).into();
                clauses.push((
                    Some(Name::Atom(ident!(record_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![atom!(size), atom_from_ident!(record.name)],
                        guards: vec![],
                        body: vec![int!(size)],
                        compiler_generated: true,
                    },
                ));
            }
            for record in module.records.values() {
                let field_names = record
                    .fields
                    .iter()
                    .fold(nil!(), |acc, f| cons!(atom_from_ident!(f.name), acc));
                clauses.push((
                    Some(Name::Atom(ident!(record_info))),
                    Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![atom!(fields), atom_from_ident!(record.name)],
                        guards: vec![],
                        body: vec![field_names],
                        compiler_generated: true,
                    },
                ));
            }

            let record_info_2 = Function {
                span: SourceSpan::UNKNOWN,
                name: ident!(record_info),
                arity: 2,
                clauses,
                spec: None,
                is_nif: false,
                var_counter: 0,
                fun_counter: 0,
            };
            define_function(module, record_info_2);
        }

        if module.callbacks.len() > 0 {
            let callbacks = module.callbacks.iter().fold(nil!(), |acc, (cbname, cb)| {
                if cb.optional {
                    acc
                } else {
                    cons!(
                        tuple!(
                            atom!(cb.span, cbname.function),
                            int!((cbname.arity as i64).into())
                        ),
                        acc
                    )
                }
            });
            let opt_callbacks = module.callbacks.iter().fold(nil!(), |acc, (cbname, cb)| {
                if cb.optional {
                    cons!(
                        tuple!(
                            atom!(cb.span, cbname.function),
                            int!((cbname.arity as i64).into())
                        ),
                        acc
                    )
                } else {
                    acc
                }
            });

            let behaviour_info_1 = fun!(behaviour_info
                                        (atom!(callbacks)) -> callbacks;
                                        (atom!(optional_callbacks)) -> opt_callbacks);

            define_function(module, behaviour_info_1);
        }

        Ok(module)
    }
}

fn define_function(module: &mut Module, f: Function) {
    let name = FunctionName::new_local(f.name.name, f.arity);
    module.exports.insert(Span::new(f.name.span, name));
    module.functions.insert(name, f);
}
