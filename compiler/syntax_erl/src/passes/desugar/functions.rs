use std::cell::RefCell;
use std::rc::Rc;

use liblumen_diagnostics::{SourceSpan, Spanned};
use liblumen_pass::Pass;
use liblumen_syntax_core::{self as syntax_core};

use crate::ast::*;
use crate::visit::{self, VisitMut};

use super::FunctionContext;

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
        self.visit_mut_function(f)?;
        Ok(f)
    }
}
impl<'m> ExpandUnqualifiedCalls<'m> {
    pub fn new(module: &'m Module) -> Self {
        Self { module }
    }
}
impl<'m> VisitMut for ExpandUnqualifiedCalls<'m> {
    // Need to expand apply with an atom callee
    fn visit_mut_apply(&mut self, apply: &mut Apply) -> anyhow::Result<()> {
        visit::visit_mut_apply(self, apply)?;
        let arity: u8 = apply.args.len().try_into().unwrap();
        let span = apply.callee.span();
        let name = match apply.callee.as_mut() {
            Expr::Literal(Literal::Atom(ident)) => {
                let local = syntax_core::FunctionName::new_local(ident.name, arity);
                if self.module.is_local(&local) {
                    FunctionName::Resolved(Spanned::new(span, local.resolve(self.module.name())))
                } else if self.module.is_import(&local) {
                    let resolved = self.module.imports.get(&local).unwrap();
                    FunctionName::Resolved(Spanned::new(span, local.resolve(resolved.module)))
                } else {
                    // This is an unresolvable function, but we catch that elsewhere
                    return Ok(());
                }
            }
            Expr::Remote(remote) => {
                if let Ok(name) = remote.try_eval(arity) {
                    FunctionName::Resolved(Spanned::new(span, name))
                } else {
                    return Ok(());
                }
            }
            _ => return Ok(()),
        };

        apply.callee = Box::new(Expr::FunctionName(name));

        Ok(())
    }

    fn visit_mut_function_name(&mut self, name: &mut FunctionName) -> anyhow::Result<()> {
        // We only care about partially-resolved function names at this point
        if let Some(ref local) = name.partial_resolution() {
            if self.module.is_local(local) {
                let span = local.span();
                let function = local.function;
                let arity = local.arity;
                *name = FunctionName::Resolved(Spanned::new(
                    span,
                    syntax_core::FunctionName::new(self.module.name(), function, arity),
                ));
            } else if self.module.is_import(local) {
                let span = local.span();
                let function = local.function;
                let arity = local.arity;
                let resolved = self.module.imports.get(local).unwrap();
                *name = FunctionName::Resolved(Spanned::new(
                    span,
                    syntax_core::FunctionName::new(resolved.module, function, arity),
                ));
            }
        }

        Ok(())
    }
}

/// This pass visits each function, and for every anonymous function in its body,
/// generates a name for that function. This results in an AST where the following
/// are true:
///
/// * All Fun nodes have been assigned a name
///
/// This pass does _not_ deal with extracting free variables, analyzing control flow, or lifting closures
/// to top-level functions, those are dealt with in other passes.
#[derive(Debug)]
pub struct NameAnonymousClosures {
    // Tracks the current top-level function we are in
    current_function: Rc<RefCell<FunctionContext>>,
    // The index of the current anonymous function. Is incremented during post-traversal
    index: usize,
}
impl NameAnonymousClosures {
    pub fn new(current_function: Rc<RefCell<FunctionContext>>) -> Self {
        Self {
            current_function,
            index: 0,
        }
    }

    fn get_name(&mut self, span: SourceSpan) -> Ident {
        let current_function = self.current_function.borrow();
        // NOTE: erlc uses the top-level function name in all anonymous function names
        // Recursive anonymous functions are treated as if they have no name
        let symbol_name = format!(
            "-{}/{}-fun-{}",
            current_function.name.as_str(),
            &current_function.arity,
            self.index
        );
        let symbol = Symbol::intern(&symbol_name);
        Ident::new(symbol, span)
    }
}
impl Pass for NameAnonymousClosures {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        self.visit_mut_function(f)?;
        Ok(f)
    }
}
impl VisitMut for NameAnonymousClosures {
    fn visit_mut_fun(&mut self, fun: &mut Fun) -> anyhow::Result<()> {
        match fun {
            Fun::Recursive(ref mut f) => {
                // We want to index bottom-up, just like erlc, so visit the function body first
                visit::visit_mut_recursive_fun(self, f)?;
                // On the way back up, we bump the index as this is an anonymous function
                self.index += 1;
                // Set the function name
                f.name.replace(self.get_name(f.span.clone()));
            }
            Fun::Anonymous(ref mut f) => {
                visit::visit_mut_anonymous_fun(self, f)?;
                self.index += 1;
                f.name.replace(self.get_name(f.span.clone()));
            }
        }

        Ok(())
    }
}
