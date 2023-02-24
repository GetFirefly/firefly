use core::ops::ControlFlow;

use firefly_diagnostics::{CodeMap, SourceSpan, Spanned};
use firefly_intern::{symbols, Ident, Symbol};
use firefly_number::Int;
use firefly_pass::Pass;

use crate::ast::*;
use crate::lexer::DelayedSubstitution;
use crate::visit::{self as visit, VisitMut};

/// This pass expands all delayed macro substitutions to their corresponding terms.
///
/// After this pass completes, the following is true of the AST:
///
/// * All `Expr::DelayedSubstitution` nodes have been replaced with `Expr::Literal` nodes
#[derive(Debug)]
pub struct ExpandSubstitutions<'cm> {
    module: Ident,
    codemap: &'cm CodeMap,
}
impl<'cm> ExpandSubstitutions<'cm> {
    pub fn new(module: Ident, codemap: &'cm CodeMap) -> Self {
        Self { module, codemap }
    }
}
impl<'cm> Pass for ExpandSubstitutions<'cm> {
    type Input<'a> = &'a mut Function;
    type Output<'a> = &'a mut Function;

    fn run<'a>(&mut self, f: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut visitor = ExpandSubstitutionsVisitor::new(self.module, self.codemap, f);
        match visitor.visit_mut_function(f) {
            ControlFlow::Continue(_) => Ok(f),
            ControlFlow::Break(err) => Err(err),
        }
    }
}

struct ExpandSubstitutionsVisitor<'cm> {
    codemap: &'cm CodeMap,
    module: Ident,
    name: Ident,
    arity: u8,
}
impl<'cm> ExpandSubstitutionsVisitor<'cm> {
    fn new(module: Ident, codemap: &'cm CodeMap, f: &Function) -> Self {
        Self {
            codemap,
            module,
            name: f.name,
            arity: f.arity,
        }
    }

    fn fix(&mut self, span: &SourceSpan, sub: DelayedSubstitution) -> Expr {
        // It shouldn't be possible for this expression to exist outside of a function context
        match sub {
            DelayedSubstitution::Module => Expr::Literal(Literal::Atom(self.module)),
            DelayedSubstitution::ModuleString => Expr::Literal(Literal::String(self.module)),
            DelayedSubstitution::FunctionName => Expr::Literal(Literal::Atom(self.name)),
            DelayedSubstitution::FunctionArity => {
                Expr::Literal(Literal::Integer(*span, self.arity.into()))
            }
            DelayedSubstitution::File => {
                let span = *span;
                if let Ok(filename) = self.codemap.name_for_span(span) {
                    let file = Ident::new(Symbol::intern(&filename.to_string()), span);
                    Expr::Literal(Literal::String(file))
                } else {
                    Expr::Literal(Literal::String(Ident::new(symbols::Empty, span)))
                }
            }
            DelayedSubstitution::Line => {
                let span = *span;
                if let Ok(loc) = self.codemap.location_for_span(span) {
                    let line = loc.line.number().to_usize() as i64;
                    Expr::Literal(Literal::Integer(span, Int::Small(line)))
                } else {
                    Expr::Literal(Literal::Integer(span, Int::Small(0)))
                }
            }
        }
    }
}
impl<'cm> VisitMut<anyhow::Error> for ExpandSubstitutionsVisitor<'cm> {
    fn visit_mut_expr(&mut self, expr: &mut Expr) -> ControlFlow<anyhow::Error> {
        if let Expr::DelayedSubstitution(sub) = expr {
            *expr = self.fix(&sub.span(), sub.item);
            ControlFlow::Continue(())
        } else {
            visit::visit_mut_expr(self, expr)
        }
    }

    fn visit_mut_pattern(&mut self, pattern: &mut Expr) -> ControlFlow<anyhow::Error> {
        if let Expr::DelayedSubstitution(sub) = pattern {
            *pattern = self.fix(&sub.span(), sub.item);
            ControlFlow::Continue(())
        } else {
            visit::visit_mut_pattern(self, pattern)
        }
    }
}
