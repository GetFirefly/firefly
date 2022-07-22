use std::collections::btree_map::Entry;

use liblumen_diagnostics::*;
use liblumen_intern::symbols;
use liblumen_syntax_core as syntax_core;

use crate::ast::*;

use super::*;

impl SemanticAnalysis {
    pub(super) fn analyze_function(&mut self, module: &mut Module, mut function: Function) {
        let resolved_name =
            syntax_core::FunctionName::new(module.name(), function.name.name, function.arity);
        let local_resolved_name = resolved_name.to_local();
        let warn_missing_specs = module
            .compile
            .as_ref()
            .map(|c| c.warn_missing_spec)
            .unwrap_or(false);

        if let Some(spec) = module.specs.get(&resolved_name) {
            function.spec.replace(spec.clone());
        } else if warn_missing_specs {
            self.reporter.show_warning(
                "missing function spec",
                &[(function.span, "expected type spec for this function")],
            );
        }

        let nif_name = Span::new(function.name.span, local_resolved_name.clone());
        if module.nifs.contains(&nif_name) {
            function.is_nif = true;
        } else {
            // Determine if the body of the function implicitly defines this as a NIF
            // Such a function will have a single clause consisting only of a call to `erlang:nif_error/1`
            if function.clauses.len() == 1 {
                let clause = &function.clauses[0].1;
                if clause.body.len() == 1 {
                    if let Expr::Apply(Apply { callee, args, .. }) = &clause.body[0] {
                        if args.len() == 1 {
                            if let Expr::Remote(remote) = callee.as_ref() {
                                if let Ok(name) = remote.try_eval(1) {
                                    if name.module == Some(symbols::Erlang)
                                        && name.function == symbols::NifError
                                    {
                                        function.is_nif = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        match module.functions.entry(local_resolved_name) {
            Entry::Vacant(f) => {
                f.insert(function);
            }
            Entry::Occupied(initial_def) => {
                let def = initial_def.into_mut();
                self.reporter.show_error(
                    "clauses from the same function should be grouped together",
                    &[
                        (function.span, "found more clauses here"),
                        (def.span, "function is first defined here"),
                    ],
                );
                def.clauses.append(&mut function.clauses);
            }
        }
    }
}
