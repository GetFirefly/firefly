use std::collections::btree_map::Entry;

use liblumen_syntax_core::{self as syntax_core};

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
            self.show_warning(
                "missing function spec",
                &[(function.span, "expected type spec for this function")],
            );
        }

        match module.functions.entry(local_resolved_name) {
            Entry::Vacant(f) => {
                f.insert(function);
            }
            Entry::Occupied(initial_def) => {
                let def = initial_def.into_mut();
                self.show_error(
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
