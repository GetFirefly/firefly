use core::ops::ControlFlow;
use std::collections::btree_map::Entry;

use firefly_intern::symbols;
use firefly_syntax_base::*;
use firefly_util::diagnostics::*;

use crate::ast::*;
use crate::visit::VisitMut;

pub fn analyze_function(
    diagnostics: &DiagnosticsHandler,
    module: &mut Module,
    mut function: Function,
) {
    let resolved_name = FunctionName::new(module.name(), function.name.name, function.arity);
    let local_resolved_name = resolved_name.to_local();
    let warn_missing_specs = module
        .compile
        .as_ref()
        .map(|c| c.warn_missing_spec)
        .unwrap_or(false);

    if let Some(spec) = module.specs.get(&resolved_name) {
        function.spec.replace(spec.clone());
    } else if warn_missing_specs {
        diagnostics
            .diagnostic(Severity::Warning)
            .with_message("missing function spec")
            .with_primary_label(function.span, "expected type spec for this function")
            .emit();
    }

    let nif_name = Span::new(function.name.span, local_resolved_name.clone());
    if module.nifs.contains(&nif_name) {
        function.is_nif = true;
    } else {
        // Determine if the body of the function implicitly defines this as a NIF
        // Such a function will have a call to erlang:nif_error/1 or /2
        let mut is_nif = IsNifVisitor;
        if let ControlFlow::Break(true) = is_nif.visit_mut_function(&mut function) {
            module.nifs.insert(nif_name);
            function.is_nif = true;
        }
    }

    // If we have a local with the same name as an imported function, the import is shadowed
    if module.imports.contains_key(&local_resolved_name) {
        module.imports.remove(&local_resolved_name);
    }

    match module.functions.entry(local_resolved_name) {
        Entry::Vacant(f) => {
            f.insert(function);
        }
        Entry::Occupied(initial_def) => {
            let def = initial_def.into_mut();
            diagnostics
                .diagnostic(Severity::Error)
                .with_message("clauses from the same function should be grouped together")
                .with_primary_label(function.span, "found more clauses here")
                .with_secondary_label(def.span, "function is first defined here")
                .emit();
            def.clauses.append(&mut function.clauses);
        }
    }
}

struct IsNifVisitor;
impl VisitMut<bool> for IsNifVisitor {
    fn visit_mut_apply(&mut self, apply: &mut Apply) -> ControlFlow<bool> {
        match apply.callee.as_ref() {
            Expr::Remote(Remote {
                module, function, ..
            }) => match (module.as_atom_symbol(), function.as_atom_symbol()) {
                (Some(symbols::Erlang), Some(symbols::NifError)) => ControlFlow::Break(true),
                _ => return ControlFlow::Continue(()),
            },
            Expr::FunctionVar(name)
                if name.module() == Some(symbols::Erlang)
                    && name.function() == Some(symbols::NifError) =>
            {
                ControlFlow::Break(true)
            }
            _ => ControlFlow::Continue(()),
        }
    }

    // As an optimization we can skip over patterns, since the calls we're looking for will only occur in a non-pattern context
    fn visit_mut_pattern(&mut self, _pattern: &mut Expr) -> ControlFlow<bool> {
        ControlFlow::Continue(())
    }

    // Likewise, we can skip over the parts of clauses we don't need to check
    fn visit_mut_clause(&mut self, clause: &mut Clause) -> ControlFlow<bool> {
        for expr in clause.body.iter_mut() {
            self.visit_mut_expr(expr)?;
        }
        ControlFlow::Continue(())
    }

    // The call to nif_error should exist near the entry of the function, so we can skip certain expression types
    // which we know the call to nif_error/2 will not be in, or which are not going to be present in a nif stub
    fn visit_mut_expr(&mut self, expr: &mut Expr) -> ControlFlow<bool> {
        match expr {
            Expr::Begin(ref mut begin) => self.visit_mut_begin(begin),
            Expr::Apply(ref mut apply) => self.visit_mut_apply(apply),
            Expr::Match(ref mut expr) => self.visit_mut_match(expr),
            Expr::If(ref mut expr) => self.visit_mut_if(expr),
            Expr::Catch(ref mut expr) => self.visit_mut_catch(expr),
            Expr::Case(ref mut case) => self.visit_mut_case(case),
            Expr::Try(ref mut expr) => self.visit_mut_try(expr),
            Expr::Protect(ref mut protect) => self.visit_mut_protect(protect),
            _ => ControlFlow::Continue(()),
        }
    }
}
