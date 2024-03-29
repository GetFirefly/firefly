use core::ops::ControlFlow;

use crate::ast::*;

use firefly_diagnostics::*;
use firefly_pass::Pass;
use firefly_syntax_base::FunctionName;

use crate::visit::{self as visit, VisitMut};

/// This pass expands partially-resolved function names to their fully-qualified
/// names, when those names refer to imported functions. It does not expand calls
/// to local functions, nor does it examine unresolved calls, as we deal with those
/// much later during compilation after constant propagation and with proper dataflow
/// analysis in place to allow us to better resolve the operands of function applications.
///
/// Once this pass has run, we don't need to concern ourselves with imports anymore, as
/// the distinction is erased.
pub struct ExpandUnqualifiedCalls<'m> {
    module: &'m Module,
}
impl<'m> Pass for ExpandUnqualifiedCalls<'m> {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        match self.visit_mut_function(f) {
            ControlFlow::Continue(_) => Ok(f),
            ControlFlow::Break(err) => Err(err),
        }
    }
}
impl<'m> ExpandUnqualifiedCalls<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self { module }
    }
}
impl<'m> VisitMut<anyhow::Error> for ExpandUnqualifiedCalls<'m> {
    // Need to expand apply with an atom callee
    fn visit_mut_apply(&mut self, apply: &mut Apply) -> ControlFlow<anyhow::Error> {
        visit::visit_mut_apply(self, apply)?;
        let arity: u8 = apply.args.len().try_into().unwrap();
        let span = apply.callee.span();
        let name = match apply.callee.as_mut() {
            Expr::Literal(Literal::Atom(ident)) => {
                let local = FunctionName::new_local(ident.name, arity);
                if self.module.is_local(&local) {
                    FunctionVar::Resolved(Span::new(span, local.resolve(self.module.name())))
                } else if self.module.is_import(&local) {
                    let resolved = self.module.imports.get(&local).unwrap();
                    FunctionVar::Resolved(Span::new(span, local.resolve(resolved.module)))
                } else {
                    // This is an unresolvable function, but we catch that elsewhere
                    return ControlFlow::Continue(());
                }
            }
            Expr::Remote(remote) => {
                if let Ok(name) = remote.try_eval(arity) {
                    FunctionVar::Resolved(Span::new(span, name))
                } else {
                    return ControlFlow::Continue(());
                }
            }
            _ => return ControlFlow::Continue(()),
        };

        apply.callee = Box::new(Expr::FunctionVar(name));

        ControlFlow::Continue(())
    }

    fn visit_mut_function_var(&mut self, name: &mut FunctionVar) -> ControlFlow<anyhow::Error> {
        // We only care about partially-resolved function names at this point
        if let Some(ref local) = name.partial_resolution() {
            if self.module.is_local(local) {
                let span = local.span();
                let function = local.function;
                let arity = local.arity;
                *name = FunctionVar::Resolved(Span::new(
                    span,
                    FunctionName::new(self.module.name(), function, arity),
                ));
            } else if self.module.is_import(local) {
                let span = local.span();
                let function = local.function;
                let arity = local.arity;
                let resolved = self.module.imports.get(local).unwrap();
                *name = FunctionVar::Resolved(Span::new(
                    span,
                    FunctionName::new(resolved.module, function, arity),
                ));
            }
        }

        ControlFlow::Continue(())
    }
}
