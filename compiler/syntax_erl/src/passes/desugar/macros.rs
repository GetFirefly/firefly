use std::cell::RefCell;
use std::rc::Rc;

use liblumen_diagnostics::SourceSpan;
use liblumen_pass::Pass;

use super::FunctionContext;
use crate::ast::*;
use crate::lexer::DelayedSubstitution;
use crate::visit::{self, VisitMut};

/// This pass expands all delayed macro substitutions to their corresponding terms.
///
/// After this pass completes, the following is true of the AST:
///
/// * All `Expr::DelayedSubstitution` nodes have been replaced with `Expr::Literal` nodes
#[derive(Debug)]
pub struct ExpandSubstitutions {
    current_function: Rc<RefCell<FunctionContext>>,
}
impl ExpandSubstitutions {
    pub fn new(current_function: Rc<RefCell<FunctionContext>>) -> Self {
        Self { current_function }
    }

    fn fix(&mut self, span: &SourceSpan, sub: DelayedSubstitution) -> Expr {
        // It shouldn't be possible for this expression to exist outside of a function context
        let current_function = self.current_function.borrow();
        match sub {
            DelayedSubstitution::FunctionName => {
                Expr::Literal(Literal::Atom(current_function.name))
            }
            DelayedSubstitution::FunctionArity => Expr::Literal(Literal::Integer(
                span.clone(),
                current_function.arity.into(),
            )),
        }
    }
}
impl Pass for ExpandSubstitutions {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        self.visit_mut_function(f)?;
        Ok(f)
    }
}
impl VisitMut for ExpandSubstitutions {
    fn visit_mut_expr(&mut self, expr: &mut Expr) -> anyhow::Result<()> {
        if let Expr::DelayedSubstitution(ref span, sub) = expr {
            *expr = self.fix(span, *sub);
            Ok(())
        } else {
            visit::visit_mut_expr(self, expr)
        }
    }

    fn visit_mut_pattern(&mut self, pattern: &mut Expr) -> anyhow::Result<()> {
        if let Expr::DelayedSubstitution(ref span, sub) = pattern {
            *pattern = self.fix(span, *sub);
            Ok(())
        } else {
            visit::visit_mut_pattern(self, pattern)
        }
    }
}
