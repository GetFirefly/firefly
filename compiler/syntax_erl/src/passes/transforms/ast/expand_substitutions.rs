use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Ident;
use liblumen_pass::Pass;

use crate::ast::*;
use crate::lexer::DelayedSubstitution;
use crate::visit::ast::{self as visit, VisitMut};

/// This pass expands all delayed macro substitutions to their corresponding terms.
///
/// After this pass completes, the following is true of the AST:
///
/// * All `Expr::DelayedSubstitution` nodes have been replaced with `Expr::Literal` nodes
#[derive(Debug)]
pub struct ExpandSubstitutions;
impl Pass for ExpandSubstitutions {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut visitor = ExpandSubstitutionsVisitor::new(f);
        visitor.visit_mut_function(f)?;
        Ok(f)
    }
}

struct ExpandSubstitutionsVisitor {
    name: Ident,
    arity: u8,
}
impl ExpandSubstitutionsVisitor {
    fn new(f: &Function) -> Self {
        Self {
            name: f.name,
            arity: f.arity,
        }
    }

    fn fix(&mut self, span: &SourceSpan, sub: DelayedSubstitution) -> Expr {
        // It shouldn't be possible for this expression to exist outside of a function context
        match sub {
            DelayedSubstitution::FunctionName => Expr::Literal(Literal::Atom(self.name)),
            DelayedSubstitution::FunctionArity => {
                Expr::Literal(Literal::Integer(span.clone(), self.arity.into()))
            }
        }
    }
}
impl VisitMut for ExpandSubstitutionsVisitor {
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
