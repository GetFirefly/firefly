use liblumen_diagnostics::*;
use liblumen_pass::Pass;
use liblumen_syntax_base::*;

use crate::ast::*;

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
pub struct DefinePseudoLocals;
impl Pass for DefinePseudoLocals {
    type Input<'a> = &'a mut Module;
    type Output<'a> = &'a mut Module;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let span = module.span;
        let mod_info_0 = fun!(module_info () -> apply!(span, erlang, get_module_info, (atom_from_ident!(module.name))));
        let mod_info_1 = fun!(module_info (Key) -> apply!(span, erlang, get_module_info, (atom_from_ident!(module.name), var!(Key))));

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

        define_function(module, mod_info_0);
        define_function(module, mod_info_1);

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
    module.functions.insert(name, f);
}
