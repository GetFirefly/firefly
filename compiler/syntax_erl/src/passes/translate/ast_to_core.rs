///! Purpose : Transform normal Erlang to Core Erlang
///!
///! At this stage all preprocessing has been done. All that is left are
///! "pure" Erlang functions.
///!
///! Core transformation is done in four stages:
///!
///! 1. Flatten expressions into an internal core form without doing
///!    matching.
///!
///! 2. Step "forwards" over the icore code annotating each "top-level"
///!    thing with variable usage.  Detect bound variables in matching
///!    and replace with explicit guard test.  Annotate "internal-core"
///!    expressions with variables they use and create.  Convert matches
///!    to cases when not pure assignments.
///!
///! 3. Step "backwards" over icore code using variable usage
///!    annotations to change implicit exported variables to explicit
///!    returns.
///!
///! 4. Lower receives to more primitive operations.  Split binary
///!    patterns where a value is matched out and then used used as
///!    a size in the same pattern.  That simplifies the subsequent
///!    passes as all variables are within a single pattern are either
///!    new or used, but never both at the same time.
///!
///! To ensure the evaluation order we ensure that all arguments are
///! safe.  A "safe" is basically a core_lib simple with VERY restricted
///! binaries.
///!
///! We have to be very careful with matches as these create variables.
///! While we try not to flatten things more than necessary we must make
///! sure that all matches are at the top level.  For this we use the
///! type "novars" which are non-match expressions.  Cases and receives
///! can also create problems due to exports variables so they are not
///! "novars" either.  I.e. a novars will not export variables.
///!
///! Annotations in the #iset, #iletrec, and all other internal records
///! is kept in a record, #a, not in a list as in proper core.  This is
///! easier and faster and creates no problems as we have complete control
///! over all annotations.
///!
///! On output, the annotation for most Core Erlang terms will contain
///! the source line number. A few terms will be marked with the atom
///! atom 'compiler_generated', to indicate that the compiler has generated
///! them and that no warning should be generated if they are optimized
///! away.
///!
///!
///! In this translation:
///!
///! call ops are safes
///! call arguments are safes
///! match arguments are novars
///! case arguments are novars
///! receive timeouts are novars
///! binaries and maps are novars
///! let/set arguments are expressions
///! fun is not a safe
use std::cell::UnsafeCell;
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use firefly_binary::{BinaryEntrySpecifier, BitVec};
use firefly_intern::{symbols, Ident, Symbol};
use firefly_number::Int;
use firefly_pass::Pass;
use firefly_syntax_base::*;
use firefly_syntax_core::passes::{
    AnnotateVariableUsage, FunctionContext, RewriteExports, RewriteReceivePrimitives,
};
use firefly_syntax_core::*;
use firefly_util::diagnostics::*;

use crate::ast;
use crate::evaluator;

use anyhow::bail;

const COLLAPSE_MAX_SIZE_SEGMENT: usize = 1024;

/// This pass transforms an AST function into its Core IR form for further analysis and eventual lowering to Kernel IR
///
/// This pass performs numerous small transformations to normalize the structure of the AST
pub struct AstToCore {
    diagnostics: Arc<DiagnosticsHandler>,
}
impl AstToCore {
    pub fn new(diagnostics: Arc<DiagnosticsHandler>) -> Self {
        Self { diagnostics }
    }
}
impl Pass for AstToCore {
    type Input<'a> = ast::Module;
    type Output<'a> = Module;

    fn run<'a>(&mut self, mut ast: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut module = Module {
            span: ast.span,
            annotations: Annotations::default(),
            name: ast.name,
            compile: ast.compile.unwrap_or_default(),
            on_load: ast.on_load,
            nifs: ast.nifs,
            exports: ast.exports,
            functions: BTreeMap::new(),
        };

        while let Some((name, function)) = ast.functions.pop_first() {
            let span = function.span();
            let spanned_name = Span::new(span, name);
            let is_nif = module.nifs.contains(&spanned_name);
            let context = Rc::new(UnsafeCell::new(FunctionContext::new(
                span,
                name,
                function.var_counter,
                function.fun_counter,
                is_nif,
            )));

            let mut pipeline = TranslateAst::new(&self.diagnostics, Rc::clone(&context))
                .chain(AnnotateVariableUsage::new(Rc::clone(&context)))
                .chain(RewriteExports::new(Rc::clone(&context)))
                .chain(RewriteReceivePrimitives::new(Rc::clone(&context)));
            let fun = pipeline.run(function)?;
            let function = Function {
                var_counter: unsafe { &*context.get() }.var_counter,
                fun,
            };
            module.functions.insert(name, function);
        }

        Ok(module)
    }
}

/// Phase 1: Lower AST to CST
///
/// This phase flattens expressions into an internal core form
/// without doing matching. This form is more amenable to further
/// transformations and lowering.
struct TranslateAst<'p> {
    diagnostics: &'p DiagnosticsHandler,
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl<'p> TranslateAst<'p> {
    fn new(diagnostics: &'p DiagnosticsHandler, context: Rc<UnsafeCell<FunctionContext>>) -> Self {
        Self {
            diagnostics,
            context,
        }
    }

    #[inline(always)]
    fn context(&self) -> &FunctionContext {
        unsafe { &*self.context.get() }
    }

    #[inline(always)]
    fn context_mut(&self) -> &mut FunctionContext {
        unsafe { &mut *self.context.get() }
    }
}
impl<'p> Pass for TranslateAst<'p> {
    type Input<'a> = ast::Function;
    type Output<'a> = IFun;

    fn run<'a>(&mut self, mut fun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Generate new variables for the function head
        let (name, span, arity, is_nif, params) = {
            let context = self.context_mut();
            let span = context.span;
            let name = Ident::new(context.name.function, span);
            let arity = context.name.arity as usize;
            let params = (0..arity).map(|_| context.next_var(Some(span))).collect();
            (name, span, arity, context.is_nif, params)
        };

        let mut annotations = Annotations::default();
        if is_nif {
            annotations.set(symbols::Nif);
        }

        // Canonicalize all of the clauses
        let mut clauses = Vec::with_capacity(fun.clauses.len());
        for (_, clause) in fun.clauses.drain(..) {
            clauses.push(self.clause(clause)?);
        }

        // Create fallback clause to handle pattern failure
        let fail = {
            let params = (0..arity)
                .map(|_| IExpr::Var(self.context_mut().next_var(Some(span))))
                .collect::<Vec<_>>();

            let reason = {
                let mut elements = vec![iatom!(span, symbols::FunctionClause)];
                elements.extend(params.iter().cloned());
                IExpr::Tuple(ITuple::new(span, elements))
            };
            fail_clause(span, params, reason)
        };

        Ok(IFun {
            span,
            annotations,
            id: Some(name),
            name: None,
            vars: params,
            clauses,
            fail,
        })
    }
}
impl<'p> TranslateAst<'p> {
    fn clauses(&mut self, mut clauses: Vec<ast::Clause>) -> anyhow::Result<Vec<IClause>> {
        let mut out = Vec::with_capacity(clauses.len());
        for clause in clauses.drain(..) {
            out.push(self.clause(clause)?);
        }
        Ok(out)
    }

    fn clause(&mut self, clause: ast::Clause) -> anyhow::Result<IClause> {
        let span = clause.span;
        match self.pattern_list(clause.patterns.clone()) {
            Ok(patterns) => {
                let guards = self.guard(clause.guards);
                let body = self.exprs(clause.body)?;
                Ok(IClause {
                    span,
                    annotations: Annotations::default(),
                    patterns,
                    guards,
                    body,
                })
            }
            Err(_) => {
                // The function head pattern can't possibly match
                // To ensure we can proceed with compilation, we rewrite the pattern
                // to a pattern that binds the same variables, but ensuring the clause is never
                // executed by having the guard return false
                self.diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("this clause can never match")
                    .with_primary_label(span, "the pattern in this clause can never succeed")
                    .emit();
                let patterns = clause.patterns.iter().cloned().map(sanitize).collect();
                assert_ne!(&clause.patterns, &patterns);
                self.clause(ast::Clause {
                    span,
                    patterns,
                    guards: vec![ast::Guard {
                        span,
                        conditions: vec![atom!(span, symbols::False)],
                    }],
                    body: clause.body,
                    compiler_generated: false,
                })
            }
        }
    }

    fn pattern_list(&mut self, mut patterns: Vec<ast::Expr>) -> anyhow::Result<Vec<IExpr>> {
        let mut output = Vec::with_capacity(patterns.len());

        for pat in patterns.drain(..) {
            output.push(self.pattern(pat)?);
        }

        Ok(output)
    }

    fn pattern(&mut self, pattern: ast::Expr) -> anyhow::Result<IExpr> {
        match pattern {
            ast::Expr::Var(ast::Var(id)) => Ok(IExpr::Var(Var::new(id))),
            ast::Expr::Literal(literal) => Ok(IExpr::Literal(literal.into())),
            ast::Expr::Cons(cons) => {
                let head = self.pattern(*cons.head)?;
                let tail = self.pattern(*cons.tail)?;
                Ok(IExpr::Cons(ICons::new(cons.span, head, tail)))
            }
            ast::Expr::Tuple(mut tuple) => {
                let mut elements = Vec::with_capacity(tuple.elements.len());
                for element in tuple.elements.drain(..) {
                    elements.push(self.pattern(element)?);
                }
                Ok(IExpr::Tuple(ITuple::new(tuple.span, elements)))
            }
            ast::Expr::Map(map) => {
                let pairs = self.pattern_map_pairs(map.fields)?;
                Ok(IExpr::Map(IMap::new_pattern(map.span, pairs)))
            }
            ast::Expr::Binary(bin) => {
                let span = bin.span();
                let segments = self.pattern_bin(bin)?;
                Ok(IExpr::Binary(IBinary::new(span, segments)))
            }
            ast::Expr::Match(ast::Match {
                box pattern,
                box expr,
                ..
            }) => {
                let p1 = self.pattern(pattern)?;
                let p2 = self.pattern(expr)?;
                self.pattern_alias(p1, p2)
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs }) => {
                if let BinaryOp::Append = op {
                    match *lhs {
                        ast::Expr::Literal(ast::Literal::Nil(_)) => self.pattern(*rhs),
                        ast::Expr::Literal(s @ ast::Literal::String(_)) => {
                            let mut elements = s.as_proper_list().unwrap();
                            let cons = elements.drain(..).rfold(rhs, |tail, lit| {
                                let span = lit.span();
                                Box::new(ast::Expr::Cons(ast::Cons {
                                    span,
                                    head: Box::new(ast::Expr::Literal(lit)),
                                    tail,
                                }))
                            });
                            self.pattern(*cons)
                        }
                        ast::Expr::Cons(ast::Cons { span, head, tail }) => {
                            let tail = Box::new(ast::Expr::BinaryExpr(ast::BinaryExpr {
                                span,
                                op: BinaryOp::Append,
                                lhs: tail,
                                rhs,
                            }));
                            self.pattern(ast::Expr::Cons(ast::Cons { span, head, tail }))
                        }
                        lhs => {
                            let expr = ast::Expr::BinaryExpr(ast::BinaryExpr {
                                span,
                                op,
                                lhs: Box::new(lhs),
                                rhs,
                            });
                            let lit = evaluator::eval_expr(&expr, None)?;
                            self.pattern(ast::Expr::Literal(lit))
                        }
                    }
                } else {
                    let expr = ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs });
                    let lit = evaluator::eval_expr(&expr, None)?;
                    self.pattern(ast::Expr::Literal(lit))
                }
            }
            expr @ ast::Expr::UnaryExpr(_) => {
                let lit = evaluator::eval_expr(&expr, None)?;
                self.pattern(ast::Expr::Literal(lit))
            }
            other => unimplemented!("{:?}", &other),
        }
    }

    fn pattern_map_pairs(
        &mut self,
        mut fields: Vec<ast::MapField>,
    ) -> anyhow::Result<Vec<IMapPair>> {
        let mut pairs = Vec::with_capacity(fields.len());
        for field in fields.drain(..) {
            if let ast::MapField::Exact { key, value, .. } = field {
                let key = match evaluator::eval_expr(&key, None) {
                    Ok(lit) => vec![IExpr::Literal(lit.into())],
                    Err(_) => self.exprs(vec![key])?,
                };
                let value = self.pattern(value)?;
                pairs.push(IMapPair {
                    op: MapOp::Exact,
                    key,
                    value: Box::new(value),
                });
            } else {
                self.diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid map pattern")
                    .with_primary_label(field.span(), "only := is permitted in map patterns")
                    .emit();
                bail!("invalid map pattern");
            }
        }
        self.pattern_alias_map_pairs(pairs)
    }

    fn pattern_alias_map_pairs(
        &mut self,
        mut pairs: Vec<IMapPair>,
    ) -> anyhow::Result<Vec<IMapPair>> {
        use std::collections::btree_map::Entry;

        let mut d0: BTreeMap<MapSortKey, Vec<IMapPair>> = BTreeMap::new();
        for pair in pairs.drain(..) {
            let k = map_sort_key(pair.key.as_slice(), &d0);
            match d0.entry(k) {
                Entry::Occupied(mut entry) => {
                    entry.get_mut().push(pair);
                }
                Entry::Vacant(entry) => {
                    entry.insert(vec![pair]);
                }
            }
        }

        let mut aliased_pairs = Vec::with_capacity(d0.len());
        for mut aliased in d0.into_values() {
            let IMapPair { op, key, value: v0 } = aliased.pop().unwrap();
            let value = aliased.drain(..).try_fold(v0, |pat, p| {
                self.pattern_alias(*p.value, *pat).map(Box::new)
            })?;
            aliased_pairs.push(IMapPair { op, key, value });
        }

        Ok(aliased_pairs)
    }

    /// Normalize aliases. Trap bad aliases by returning Err
    fn pattern_alias(&mut self, p1: IExpr, p2: IExpr) -> anyhow::Result<IExpr> {
        match (p1, p2) {
            (IExpr::Var(v1), IExpr::Var(v2)) if v1 == v2 => Ok(IExpr::Var(v1)),
            (IExpr::Var(v1), IExpr::Alias(mut alias)) => {
                if v1 == alias.var {
                    Ok(IExpr::Alias(alias))
                } else {
                    alias.pattern = Box::new(self.pattern_alias(IExpr::Var(v1), *alias.pattern)?);
                    Ok(IExpr::Alias(alias))
                }
            }
            (IExpr::Var(v), p2) => Ok(IExpr::Alias(IAlias::new(v.span(), v, p2))),
            (IExpr::Alias(alias), IExpr::Var(v)) if alias.var == v => Ok(IExpr::Alias(alias)),
            (IExpr::Alias(a1), IExpr::Alias(a2)) => {
                let v1 = a1.var;
                let v2 = a2.var;
                let pat = self.pattern_alias(*a1.pattern, *a2.pattern)?;
                if v1 == v2 {
                    Ok(IExpr::Alias(IAlias::new(v1.span(), v1, pat)))
                } else {
                    let p2 = self.pattern_alias(IExpr::Var(v2), pat)?;
                    self.pattern_alias(IExpr::Var(v1), p2)
                }
            }
            (IExpr::Alias(alias), p2) => Ok(IExpr::Alias(IAlias::new(
                alias.span(),
                alias.var,
                self.pattern_alias(*alias.pattern, p2)?,
            ))),
            (
                IExpr::Map(IMap {
                    span,
                    annotations,
                    arg,
                    pairs: mut pairs1,
                    ..
                }),
                IExpr::Map(IMap {
                    pairs: mut pairs2, ..
                }),
            ) => {
                pairs1.append(&mut pairs2);
                Ok(IExpr::Map(IMap {
                    span,
                    annotations,
                    arg,
                    pairs: self.pattern_alias_map_pairs(pairs1)?,
                    is_pattern: true,
                }))
            }
            (p1, IExpr::Var(var)) => Ok(IExpr::Alias(IAlias::new(var.span(), var, p1))),
            (p1, IExpr::Alias(mut alias)) => {
                alias.pattern = Box::new(self.pattern_alias(p1, *alias.pattern)?);
                Ok(IExpr::Alias(alias))
            }
            (p1, p2) => {
                // Aliases between binaries are not allowed, so the only legal patterns that remain are data patterns.
                if !p1.is_data() {
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid alias pattern")
                        .with_primary_label(p1.span(), "this is not a legal pattern")
                        .emit();
                    bail!("invalid alias pattern")
                }
                if !p2.is_data() {
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid alias pattern")
                        .with_primary_label(p1.span(), "this is not a legal pattern")
                        .emit();
                    bail!("invalid alias pattern")
                }
                match (p1, p2) {
                    (IExpr::Literal(l1), IExpr::Literal(l2)) => {
                        if l1.value != l2.value {
                            self.diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("invalid alias pattern")
                                .with_primary_label(l1.span(), "this pattern cannot alias")
                                .with_secondary_label(
                                    l2.span(),
                                    "because this literal has a different value",
                                )
                                .emit();
                            bail!("invalid alias pattern")
                        }
                        Ok(IExpr::Literal(l1))
                    }
                    (
                        IExpr::Cons(ICons {
                            span,
                            head: h1,
                            tail: t1,
                            ..
                        }),
                        IExpr::Cons(ICons {
                            head: h2, tail: t2, ..
                        }),
                    ) => {
                        let head = self.pattern_alias(*h1, *h2)?;
                        let tail = self.pattern_alias(*t1, *t2)?;
                        Ok(IExpr::Cons(ICons::new(span, head, tail)))
                    }
                    (
                        IExpr::Tuple(ITuple {
                            span,
                            elements: mut es1,
                            ..
                        }),
                        IExpr::Tuple(ITuple {
                            span: span2,
                            elements: mut es2,
                            ..
                        }),
                    ) => {
                        if es1.len() != es2.len() {
                            self.diagnostics
                                .diagnostic(Severity::Error)
                                .with_message("invalid alias pattern")
                                .with_primary_label(span, "this tuple pattern cannot alias")
                                .with_secondary_label(
                                    span2,
                                    "because this tuple pattern has a different number of elements",
                                )
                                .emit();
                            bail!("invalid alias pattern")
                        }
                        let mut aliased = Vec::with_capacity(es1.len());
                        for (a1, a2) in es1.drain(..).zip(es2.drain(..)) {
                            aliased.push(self.pattern_alias(a1, a2)?);
                        }
                        Ok(IExpr::Tuple(ITuple::new(span, aliased)))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn pattern_bin(&mut self, bin: ast::Binary) -> anyhow::Result<Vec<IBitstring>> {
        let ps = self.pattern_bin_expand_strings(bin.elements);
        self.pattern_segments(ps)
    }

    fn pattern_bin_expand_strings(
        &mut self,
        mut elements: Vec<ast::BinaryElement>,
    ) -> Vec<ast::BinaryElement> {
        let mut expanded = Vec::with_capacity(elements.len());
        for element in elements.drain(..) {
            match element {
                ast::BinaryElement {
                    specifier: None,
                    bit_size: None,
                    bit_expr: ast::Expr::Literal(ast::Literal::String(s)),
                    ..
                } if s != symbols::Empty => {
                    let mut s = bin_expand_string(s, Int::Small(0), 0);
                    expanded.extend(s.drain(..));
                }
                ast::BinaryElement {
                    span,
                    specifier,
                    bit_size,
                    bit_expr: ast::Expr::Literal(ast::Literal::String(s)),
                } => {
                    for c in s.as_str().get().chars() {
                        let bit_expr = ast::Expr::Literal(ast::Literal::Char(span, c));
                        let bit_size = bit_size.clone();
                        let specifier = specifier.clone();
                        expanded.push(ast::BinaryElement {
                            span,
                            bit_expr,
                            bit_size,
                            specifier,
                        });
                    }
                }
                element => expanded.push(element),
            }
        }
        expanded
    }

    fn pattern_segments(
        &mut self,
        mut elements: Vec<ast::BinaryElement>,
    ) -> anyhow::Result<Vec<IBitstring>> {
        let mut segments = Vec::with_capacity(elements.len());
        for element in elements.drain(..) {
            segments.push(self.pattern_segment(element)?);
        }
        Ok(segments)
    }

    fn pattern_segment(&mut self, element: ast::BinaryElement) -> anyhow::Result<IBitstring> {
        let span = element.span;
        match make_bit_type(span, element.bit_size.as_ref(), element.specifier) {
            Ok((size, spec)) => {
                let value = self.pattern(element.bit_expr)?;
                let value = match spec {
                    BinaryEntrySpecifier::Float { .. } => value.coerce_to_float(),
                    _ => value,
                };
                let size = match size {
                    None => vec![],
                    Some(size) => match evaluator::eval_expr(&size, None) {
                        Ok(lit) => vec![IExpr::Literal(lit.into())],
                        Err(_) => self.exprs(vec![size])?,
                    },
                };
                Ok(IBitstring {
                    span,
                    annotations: Annotations::default(),
                    value: Box::new(value),
                    size,
                    spec,
                })
            }
            Err(reason) => {
                self.diagnostics
                    .diagnostic(Severity::Error)
                    .with_message("invalid binary element")
                    .with_primary_label(element.span, reason)
                    .with_secondary_label(span, "in this binary expression")
                    .emit();
                bail!("invalid binary pattern")
            }
        }
    }

    /// guard([Expr], State) -> {[Cexpr],State}.
    ///  Build an explicit and/or tree of guard alternatives, then traverse
    ///  top-level and/or tree and "protect" inner tests.
    fn guard(&mut self, mut guards: Vec<ast::Guard>) -> Vec<IExpr> {
        if guards.is_empty() {
            return vec![];
        }

        let last = guards.pop().unwrap();
        let guard = guards.drain(..).rfold(
            self.guard_tests(last.span, last.conditions),
            |acc, guard| {
                let span = guard.span;
                let span = span.merge(acc.span()).unwrap_or(span);
                let gt = self.guard_tests(span, guard.conditions);
                ast::Expr::BinaryExpr(ast::BinaryExpr {
                    span,
                    op: BinaryOp::Or,
                    lhs: Box::new(gt),
                    rhs: Box::new(acc),
                })
            },
        );
        self.context_mut().in_guard = true;
        let gexpr = self.gexpr_top(guard);
        self.context_mut().in_guard = false;
        gexpr
    }

    fn guard_tests(&mut self, span: SourceSpan, mut guards: Vec<ast::Expr>) -> ast::Expr {
        let last = guards.pop().unwrap();
        let body = guards.drain(..).rfold(last, |acc, guard| {
            let span = guard.span();
            let span = span.merge(acc.span()).unwrap_or(span);
            ast::Expr::BinaryExpr(ast::BinaryExpr {
                span,
                op: BinaryOp::And,
                lhs: Box::new(guard),
                rhs: Box::new(acc),
            })
        });
        ast::Expr::Protect(ast::Protect {
            span,
            body: Box::new(body),
        })
    }

    /// gexpr_top(Expr, State) -> {Cexpr,State}.
    ///  Generate an internal core expression of a guard test.  Explicitly
    ///  handle outer boolean expressions and "protect" inner tests in a
    ///  reasonably smart way.
    fn gexpr_top(&mut self, expr: ast::Expr) -> Vec<IExpr> {
        let (expr, pre, bools) = self.gexpr(expr, vec![]);
        let (expr, mut pre) = self.force_booleans(bools, expr, pre);
        pre.push(expr);
        pre
    }

    fn gexpr(&mut self, expr: ast::Expr, bools: Vec<IExpr>) -> (IExpr, Vec<IExpr>, Vec<IExpr>) {
        match expr {
            ast::Expr::Protect(ast::Protect { span, body }) => {
                let (expr, pre, bools2) = self.gexpr(*body, vec![]);
                if pre.is_empty() {
                    let (expr, pre) = self.force_booleans(bools2, expr, vec![]);
                    (expr, pre, bools)
                } else {
                    let (expr, mut pre) = self.force_booleans(bools2, expr, pre);
                    pre.push(expr);
                    (
                        IExpr::Protect(IProtect {
                            span,
                            annotations: Annotations::default(),
                            body: pre,
                        }),
                        vec![],
                        bools,
                    )
                }
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs })
                if op == BinaryOp::AndAlso =>
            {
                let var = self.context_mut().next_var(Some(span));
                let atom_false = atom!(span, false);
                let expr = make_bool_switch(span, *lhs, ast::Var(var.name), *rhs, atom_false);
                self.gexpr(expr, bools)
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs })
                if op == BinaryOp::OrElse =>
            {
                let var = self.context_mut().next_var(Some(span));
                let atom_true = atom!(span, true);
                let expr = make_bool_switch(span, *lhs, ast::Var(var.name), atom_true, *rhs);
                self.gexpr(expr, bools)
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs }) => {
                if op.is_boolean(2) {
                    self.gexpr_bool(span, op, *lhs, *rhs, bools)
                } else {
                    self.gexpr_test(
                        ast::Expr::BinaryExpr(ast::BinaryExpr { span, op, lhs, rhs }),
                        bools,
                    )
                }
            }
            ast::Expr::UnaryExpr(ast::UnaryExpr { op, operand, .. }) if op == UnaryOp::Not => {
                self.gexpr_not(*operand, bools)
            }
            ast::Expr::Apply(ast::Apply {
                span,
                callee,
                mut args,
            }) => {
                let arity = args.len().try_into().unwrap();
                match *callee {
                    ast::Expr::FunctionVar(name) => {
                        let op = match (name.module(), name.function()) {
                            (None, Some(symbols::Not)) if arity == 1 => {
                                let operand = args.pop().unwrap();
                                return self.gexpr_not(operand, bools);
                            }
                            (None, Some(function)) if arity == 2 => {
                                BinaryOp::from_symbol(function).ok()
                            }
                            (Some(symbols::Erlang), Some(symbols::Not)) if arity == 1 => {
                                let operand = args.pop().unwrap();
                                return self.gexpr_not(operand, bools);
                            }
                            (Some(symbols::Erlang), Some(function)) if arity == 2 => {
                                BinaryOp::from_symbol(function).ok()
                            }
                            _ => None,
                        };
                        if let Some(op) = op {
                            if op.is_boolean(arity) {
                                let rhs = args.pop().unwrap();
                                let lhs = args.pop().unwrap();
                                return self.gexpr_bool(span, op, lhs, rhs, bools);
                            }
                        }
                        self.gexpr_test(
                            ast::Expr::Apply(ast::Apply {
                                span,
                                callee: Box::new(ast::Expr::FunctionVar(name)),
                                args,
                            }),
                            bools,
                        )
                    }
                    ast::Expr::Remote(remote) if arity == 2 => {
                        if let Ok(name) = remote.try_eval(arity) {
                            if name.module == Some(symbols::Erlang) {
                                if let Ok(op) = BinaryOp::from_symbol(name.function) {
                                    if op.is_boolean(arity) {
                                        let rhs = args.pop().unwrap();
                                        let lhs = args.pop().unwrap();
                                        return self.gexpr_bool(span, op, lhs, rhs, bools);
                                    }
                                }
                            }
                            self.gexpr_test(
                                ast::Expr::Apply(ast::Apply {
                                    span,
                                    callee: Box::new(Span::new(remote.span, name).into()),
                                    args,
                                }),
                                bools,
                            )
                        } else {
                            self.gexpr_test(
                                ast::Expr::Apply(ast::Apply {
                                    span,
                                    callee: Box::new(ast::Expr::Remote(remote)),
                                    args,
                                }),
                                bools,
                            )
                        }
                    }
                    ast::Expr::Remote(remote) if arity == 1 => {
                        if let Ok(name) = remote.try_eval(arity) {
                            if name.module == Some(symbols::Erlang) && name.function == symbols::Not
                            {
                                let operand = args.pop().unwrap();
                                return self.gexpr_not(operand, bools);
                            }
                            self.gexpr_test(
                                ast::Expr::Apply(ast::Apply {
                                    span,
                                    callee: Box::new(Span::new(remote.span, name).into()),
                                    args,
                                }),
                                bools,
                            )
                        } else {
                            self.gexpr_test(
                                ast::Expr::Apply(ast::Apply {
                                    span,
                                    callee: Box::new(ast::Expr::Remote(remote)),
                                    args,
                                }),
                                bools,
                            )
                        }
                    }
                    callee => self.gexpr_test(
                        ast::Expr::Apply(ast::Apply {
                            span,
                            callee: Box::new(callee),
                            args,
                        }),
                        bools,
                    ),
                }
            }
            expr => self.gexpr_test(expr, bools),
        }
    }

    // gexpr_bool(L, R, Bools, State) -> {Cexpr,[PreExp],Bools,State}.
    //  Generate a guard for boolean operators
    fn gexpr_bool(
        &mut self,
        span: SourceSpan,
        op: BinaryOp,
        lhs: ast::Expr,
        rhs: ast::Expr,
        bools: Vec<IExpr>,
    ) -> (IExpr, Vec<IExpr>, Vec<IExpr>) {
        let (lexpr, mut lpre, bools) = self.gexpr(lhs, bools);
        let (lexpr, mut lpre2) = force_safe(self.context_mut(), lexpr);
        let (rexpr, mut rpre, bools) = self.gexpr(rhs, bools);
        let (rexpr, mut rpre2) = force_safe(self.context_mut(), rexpr);
        let call = IExpr::Call(ICall::new(
            span,
            symbols::Erlang,
            op.to_symbol(),
            vec![lexpr, rexpr],
        ));
        lpre.append(&mut lpre2);
        lpre.append(&mut rpre);
        lpre.append(&mut rpre2);
        (call, lpre, bools)
    }

    // gexpr_not(Expr, Bools, State) -> {Cexpr,[PreExp],Bools,State}.
    //  Generate an erlang:'not'/1 guard test.
    fn gexpr_not(&mut self, expr: ast::Expr, bools: Vec<IExpr>) -> (IExpr, Vec<IExpr>, Vec<IExpr>) {
        let (expr, mut pre, bools) = self.gexpr(expr, bools);
        let expr = match expr {
            IExpr::Call(mut call)
                if call.is_static(symbols::Erlang, symbols::EqualStrict, 1)
                    && call.args[1].as_boolean() == Some(true) =>
            {
                if call.annotations.contains(symbols::CompilerGenerated) {
                    // We here have the expression:
                    //
                    //    not(Expr =:= true)
                    //
                    // The annotations tested in the code above guarantees
                    // that the original expression in the Erlang source
                    // code was:
                    //
                    //    not Expr
                    //
                    // That expression can be transformed as follows:
                    //
                    //    not Expr  ==>  Expr =:= false
                    //
                    // which will produce the same result, but may eliminate
                    // redundant is_boolean/1 tests (see unforce/3).
                    //
                    // Note that this transformation would not be safe if the
                    // original expression had been:
                    //
                    //    not(Expr =:= true)
                    //
                    let b = call.args.pop().unwrap();
                    call.args.push(iatom!(b.span(), symbols::False));
                    let (expr, mut pre2) = force_safe(self.context_mut(), IExpr::Call(call));
                    pre.append(&mut pre2);
                    return (expr, pre, bools);
                } else {
                    IExpr::Call(call)
                }
            }
            expr => expr,
        };
        let span = expr.span();
        let (expr, mut pre2) = force_safe(self.context_mut(), expr);
        let call = IExpr::Call(ICall::new(span, symbols::Erlang, symbols::Not, vec![expr]));
        pre.append(&mut pre2);
        (call, pre, bools)
    }

    // gexpr_test(Expr, Bools, State) -> {Cexpr,[PreExp],Bools,State}.
    //  Generate a guard test.  At this stage we must be sure that we have
    //  a proper boolean value here so wrap things with an true test if we
    //  don't know, i.e. if it is not a comparison or a type test.
    fn gexpr_test(
        &mut self,
        expr: ast::Expr,
        mut bools: Vec<IExpr>,
    ) -> (IExpr, Vec<IExpr>, Vec<IExpr>) {
        match expr {
            ast::Expr::Literal(ast::Literal::Atom(a)) if a.name.is_boolean() => {
                (iatom!(a.span, a.name), vec![], bools)
            }
            expr => {
                let (expr, mut pre) = self.expr(expr).unwrap();
                // Generate "top-level" test and argument calls
                match expr {
                    IExpr::Call(call)
                        if call.is_static(symbols::Erlang, symbols::IsFunction, 2) =>
                    {
                        // is_function/2 is not a safe type test. We must force it to be protected.
                        let span = call.span;
                        let v = self.context_mut().next_var(Some(span));
                        pre.push(IExpr::Set(ISet::new(span, v.clone(), IExpr::Call(call))));
                        (icall_eq_true!(span, IExpr::Var(v)), pre, bools)
                    }
                    IExpr::Call(call)
                        if call.module.is_atom_value(symbols::Erlang)
                            && call.function.is_atom() =>
                    {
                        let function = call.function.as_atom().unwrap();
                        let arity = call.args.len().try_into().unwrap();
                        if is_new_type_test(function, arity)
                            || is_cmp_op(function, arity)
                            || is_bool_op(function, arity)
                        {
                            (IExpr::Call(call), pre, bools)
                        } else {
                            let span = call.span;
                            let v = self.context_mut().next_var(Some(span));
                            bools.insert(0, IExpr::Var(v.clone()));
                            pre.push(IExpr::Set(ISet::new(span, v.clone(), IExpr::Call(call))));
                            (icall_eq_true!(span, IExpr::Var(v)), pre, bools)
                        }
                    }
                    expr => {
                        let span = expr.span();
                        if expr.is_simple() {
                            bools.insert(0, expr.clone());
                            (icall_eq_true!(span, expr), pre, bools)
                        } else {
                            let v = self.context_mut().next_var(Some(span));
                            bools.insert(0, IExpr::Var(v.clone()));
                            pre.push(IExpr::Set(ISet {
                                span,
                                annotations: Annotations::default_compiler_generated(),
                                var: v.clone(),
                                arg: Box::new(expr),
                            }));
                            (icall_eq_true!(span, IExpr::Var(v)), pre, bools)
                        }
                    }
                }
            }
        }
    }

    // force_booleans([Var], E, Eps, St) -> Expr.
    //  Variables used in the top-level of a guard must be booleans.
    //
    //  Add necessary is_boolean/1 guard tests to ensure that the guard
    //  will fail if any of the variables is not a boolean.
    fn force_booleans(
        &mut self,
        mut vars: Vec<IExpr>,
        expr: IExpr,
        pre: Vec<IExpr>,
    ) -> (IExpr, Vec<IExpr>) {
        for var in vars.iter_mut() {
            var.annotations_mut().clear();
        }

        // Prune the list of variables that will need is_boolean/1
        // tests. Basically, if the guard consists of simple expressions
        // joined by 'and's no is_boolean/1 tests are needed.
        let vars = unforce(&expr, pre.clone(), vars);

        // Add is_boolean/1 tests for the remaining variables
        self.force_booleans_1(vars, expr, pre)
    }

    fn force_booleans_1(
        &mut self,
        mut vars: Vec<IExpr>,
        expr: IExpr,
        pre: Vec<IExpr>,
    ) -> (IExpr, Vec<IExpr>) {
        vars.drain(..).fold((expr, pre), |(expr, mut pre), var| {
            let span = expr.span();
            let (expr, mut pre2) = force_safe(self.context_mut(), expr);
            let mut call = ICall::new(span, symbols::Erlang, symbols::IsBoolean, vec![var]);
            call.annotations.set(symbols::CompilerGenerated);
            let call = IExpr::Call(call);
            let v = self.context_mut().next_var(Some(span));
            let set = IExpr::Set(ISet {
                span,
                annotations: Annotations::default(),
                var: v.clone(),
                arg: Box::new(call),
            });
            pre.append(&mut pre2);
            pre.push(set);
            let mut call = ICall::new(
                span,
                symbols::Erlang,
                symbols::And,
                vec![expr, IExpr::Var(v)],
            );
            call.annotations.set(symbols::CompilerGenerated);
            let expr = IExpr::Call(call);
            (expr, pre)
        })
    }

    fn exprs(&mut self, mut exprs: Vec<ast::Expr>) -> anyhow::Result<Vec<IExpr>> {
        let mut output = Vec::with_capacity(exprs.len());

        for expr in exprs.drain(..) {
            let (expr, mut pre) = self.expr(expr)?;
            output.append(&mut pre);
            output.push(expr);
        }

        Ok(output)
    }

    fn expr(&mut self, expr: ast::Expr) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        match expr {
            ast::Expr::Var(ast::Var(id)) => Ok((
                IExpr::Var(Var {
                    annotations: Annotations::default(),
                    name: id,
                    arity: None,
                }),
                vec![],
            )),
            ast::Expr::Literal(lit) => Ok((IExpr::Literal(lit.into()), vec![])),
            ast::Expr::Cons(ast::Cons { span, head, tail }) => {
                let (mut es, pre) = self.safe_list(span, vec![*head, *tail])?;
                let tail = es.pop().unwrap();
                let head = es.pop().unwrap();
                Ok((icons!(span, head, tail), pre))
            }
            ast::Expr::ListComprehension(ast::ListComprehension {
                span,
                body,
                qualifiers,
            }) => {
                let qualifiers = self.preprocess_quals(qualifiers)?;
                self.lc_tq(span, *body, qualifiers, inil!(span))
            }
            ast::Expr::BinaryComprehension(ast::BinaryComprehension {
                span,
                body,
                qualifiers,
            }) => {
                let qualifiers = self.preprocess_quals(qualifiers)?;
                self.bc_tq(span, *body, qualifiers)
            }
            ast::Expr::Tuple(ast::Tuple { span, elements }) => {
                let (elements, pre) = self.safe_list(span, elements)?;
                Ok((IExpr::Tuple(ITuple::new(span, elements)), pre))
            }
            ast::Expr::Map(ast::Map { span, fields }) => {
                self.map_build_pairs(span, IExpr::Literal(Literal::map(span, vec![])), fields)
            }
            ast::Expr::MapUpdate(ast::MapUpdate { span, map, updates }) => {
                self.expr_map(span, *map, updates)
            }
            ast::Expr::Binary(ast::Binary { span, elements }) => {
                match self.expr_bin(span, elements) {
                    Ok(ok) => Ok(ok),
                    Err(pre) => {
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("invalid binary expression")
                            .with_primary_label(
                                span,
                                "this binary expression has an invalid element",
                            )
                            .emit();
                        let badarg = iatom!(span, symbols::Badarg);
                        Ok((
                            IExpr::PrimOp(IPrimOp::new(span, symbols::Error, vec![badarg])),
                            pre,
                        ))
                    }
                }
            }
            ast::Expr::Begin(ast::Begin { mut body, .. }) => {
                // Inline the block directly.
                let expr = body.pop().unwrap();
                let mut exprs = self.exprs(body)?;
                let (expr, mut pre) = self.expr(expr)?;
                exprs.append(&mut pre);
                Ok((expr, exprs))
            }
            ast::Expr::If(ast::If { span, clauses }) => {
                let clauses = self.clauses(clauses)?;
                let fail = fail_clause(span, vec![], iatom!(span, symbols::IfClause));
                let case = IExpr::Case(ICase {
                    span,
                    annotations: Annotations::default(),
                    args: vec![],
                    clauses,
                    fail,
                });
                Ok((case, vec![]))
            }
            ast::Expr::Case(ast::Case {
                span,
                expr,
                clauses,
            }) => {
                let (expr, pre) = self.novars(*expr)?;
                let clauses = self.clauses(clauses)?;
                let fpat = self.context_mut().next_var(Some(span));
                let reason = ituple!(
                    span,
                    iatom!(span, symbols::CaseClause),
                    IExpr::Var(fpat.clone())
                );
                let fail = fail_clause(span, vec![IExpr::Var(fpat)], reason);
                let case = IExpr::Case(ICase {
                    span,
                    annotations: Annotations::default(),
                    args: vec![expr],
                    clauses,
                    fail,
                });
                Ok((case, pre))
            }
            ast::Expr::Receive(ast::Receive {
                span,
                clauses: Some(clauses),
                after: None,
            }) => {
                let clauses = self.clauses(clauses)?;
                let recv = IExpr::Receive1(IReceive1 {
                    span,
                    annotations: Annotations::default(),
                    clauses,
                });
                Ok((recv, vec![]))
            }
            ast::Expr::Receive(ast::Receive {
                span,
                clauses,
                after: Some(ast::After { timeout, body, .. }),
            }) => {
                let (timeout, pre) = self.novars(*timeout)?;
                let action = self.exprs(body)?;
                let clauses = self.clauses(clauses.unwrap_or_default())?;
                let recv = IExpr::Receive2(IReceive2 {
                    span,
                    annotations: Annotations::default(),
                    clauses,
                    timeout: Box::new(timeout),
                    action,
                });
                Ok((recv, pre))
            }
            // try .. catch .. end
            ast::Expr::Try(ast::Try {
                span,
                exprs,
                clauses: None,
                catch_clauses: Some(ccs),
                after: None,
            }) => {
                let exprs = self.exprs(exprs)?;
                let v = self.context_mut().next_var(Some(span));
                let (evars, handler) = self.try_exception(ccs)?;
                let texpr = IExpr::Try(ITry {
                    span,
                    annotations: Annotations::default(),
                    args: exprs,
                    vars: vec![v.clone()],
                    body: vec![IExpr::Var(v)],
                    evars,
                    handler: Box::new(handler),
                });
                Ok((texpr, vec![]))
            }
            // try .. of .. catch .. end
            ast::Expr::Try(ast::Try {
                span,
                exprs,
                clauses: Some(cs),
                catch_clauses: Some(ccs),
                after: None,
            }) => {
                let exprs = self.exprs(exprs)?;
                let v = self.context_mut().next_var(Some(span));
                let clauses = self.clauses(cs)?;
                let fpat = self.context_mut().next_var(Some(span));
                let fail = fail_clause(
                    span,
                    vec![IExpr::Var(fpat.clone())],
                    ituple!(span, iatom!(span, symbols::TryClause), IExpr::Var(fpat)),
                );
                let (evars, handler) = self.try_exception(ccs)?;
                let case = IExpr::Case(ICase {
                    span,
                    annotations: Annotations::default(),
                    args: vec![IExpr::Var(v.clone())],
                    clauses,
                    fail,
                });
                let texpr = IExpr::Try(ITry {
                    span,
                    annotations: Annotations::default(),
                    args: exprs,
                    vars: vec![v],
                    body: vec![case],
                    evars,
                    handler: Box::new(handler),
                });
                Ok((texpr, vec![]))
            }
            // try .. after .. end
            ast::Expr::Try(ast::Try {
                span,
                exprs,
                clauses: None,
                catch_clauses: None,
                after: Some(after),
            }) => self.try_after(span, exprs, after),
            // try .. [of ...] [catch ... ] after .. end
            ast::Expr::Try(ast::Try {
                span,
                exprs,
                clauses,
                catch_clauses,
                after: Some(after),
            }) => {
                let outer = ast::Expr::Try(ast::Try {
                    span,
                    exprs: vec![ast::Expr::Try(ast::Try {
                        span,
                        exprs,
                        clauses,
                        catch_clauses,
                        after: None,
                    })],
                    clauses: None,
                    catch_clauses: None,
                    after: Some(after),
                });
                self.expr(outer)
            }
            ast::Expr::Catch(ast::Catch { span, expr }) => {
                let (expr, mut pre) = self.expr(*expr)?;
                pre.push(expr);

                let cexpr = IExpr::Catch(ICatch {
                    span,
                    annotations: Annotations::default(),
                    body: pre,
                });
                Ok((cexpr, vec![]))
            }
            ast::Expr::FunctionVar(name) => {
                match name {
                    ast::FunctionVar::Resolved(name) => {
                        let span = name.span();
                        let module = iatom!(span, name.module.unwrap());
                        let function = iatom!(span, name.function);
                        let arity = iint!(span, name.arity);
                        let call = IExpr::PrimOp(IPrimOp::new(
                            span,
                            symbols::MakeFun,
                            vec![module, function, arity],
                        ));
                        Ok((call, vec![]))
                    }
                    ast::FunctionVar::PartiallyResolved(name) => {
                        // Generate a new name for eta conversion of local funs (`fun local/123`)
                        let fname = self.context_mut().new_fun_name(None);
                        let span = name.span();
                        let id = Literal::tuple(
                            span,
                            vec![
                                Literal::integer(span, 0),
                                Literal::integer(span, 0),
                                Literal::atom(span, fname),
                            ],
                        );
                        Ok((
                            IExpr::Var(Var {
                                annotations: Annotations::from([(symbols::Id, id.into())]),
                                name: Ident::new(name.function, span),
                                arity: Some(name.arity as usize),
                            }),
                            vec![],
                        ))
                    }
                    ast::FunctionVar::Unresolved(ast::UnresolvedFunctionName {
                        span,
                        module: None,
                        function,
                        arity,
                    }) => {
                        use crate::Arity;
                        // Generate a new name for eta conversion of local funs (`fun local/123`)
                        let Arity::Int(arity) = arity else { panic!("unexpected dynamic arity") };
                        let fname = self.context_mut().new_fun_name(None);
                        let id = Literal::tuple(
                            span,
                            vec![
                                Literal::integer(span, 0),
                                Literal::integer(span, 0),
                                Literal::atom(span, fname),
                            ],
                        );
                        Ok((
                            IExpr::Var(Var {
                                annotations: Annotations::from([(symbols::Id, id.into())]),
                                name: function.ident(),
                                arity: Some(arity as usize),
                            }),
                            vec![],
                        ))
                    }
                    ast::FunctionVar::Unresolved(ast::UnresolvedFunctionName {
                        span,
                        module: Some(module),
                        function,
                        arity,
                    }) => {
                        let module = module.into();
                        let function = function.into();
                        let arity = Span::new(span, arity).into();
                        let (mfa, pre) = self.safe_list(span, vec![module, function, arity])?;
                        let call = IExpr::PrimOp(IPrimOp::new(span, symbols::MakeFun, mfa));
                        Ok((call, pre))
                    }
                }
            }
            ast::Expr::Fun(ast::Fun::Recursive(mut fun)) => self.fun_tq(
                fun.span,
                Some(fun.self_name),
                fun.clauses.drain(..).map(|(_, c)| c).collect(),
            ),
            ast::Expr::Fun(ast::Fun::Anonymous(fun)) => self.fun_tq(fun.span, None, fun.clauses),
            ast::Expr::Apply(ast::Apply {
                span,
                callee,
                mut args,
            }) => match *callee {
                ast::Expr::Remote(ast::Remote {
                    span: remote_span,
                    module,
                    function,
                    ..
                }) => {
                    let mut safes = vec![*module, *function];
                    safes.append(&mut args);
                    let (mut args, pre) = self.safe_list(remote_span, safes)?;
                    let mut argv = args.split_off(2);
                    let mut mf = args;
                    let function = mf.pop().unwrap();
                    let module = mf.pop().unwrap();
                    let is_erlang = module.is_atom_value(symbols::Erlang);
                    let maybe_bif = if is_erlang {
                        match function.as_atom() {
                            Some(sym) => match sym {
                                symbols::Error
                                | symbols::Exit
                                | symbols::Throw
                                | symbols::Raise
                                | symbols::NifError => Some(sym),
                                _ => None,
                            },
                            None => None,
                        }
                    } else {
                        None
                    };
                    let is_possible_record_match_fail =
                        is_erlang && maybe_bif == Some(symbols::Error) && argv.len() == 1;
                    if is_possible_record_match_fail {
                        let arg = argv.pop().unwrap();
                        if let IExpr::Tuple(tuple) = arg {
                            if matches!(
                                &tuple.elements[0],
                                IExpr::Literal(Literal {
                                    value: Lit::Atom(symbols::Badrecord),
                                    ..
                                })
                            ) {
                                let fail = IExpr::PrimOp(IPrimOp::new(
                                    span,
                                    symbols::MatchFail,
                                    vec![IExpr::Tuple(tuple)],
                                ));
                                return Ok((fail, pre));
                            } else {
                                argv.push(IExpr::Tuple(tuple));
                            }
                        } else {
                            argv.push(arg);
                        }
                    }
                    if let Some(op) = maybe_bif {
                        let bif = IExpr::PrimOp(IPrimOp::new(span, op, argv));
                        Ok((bif, pre))
                    } else {
                        let call = IExpr::Call(ICall {
                            span,
                            annotations: Annotations::default(),
                            module: Box::new(module),
                            function: Box::new(function),
                            args: argv,
                        });
                        Ok((call, pre))
                    }
                }
                ast::Expr::FunctionVar(name) => {
                    let nspan = name.span();
                    let (m, f, _) = name.mfa();
                    let remote = ast::Expr::Remote(ast::Remote {
                        span: nspan,
                        module: Box::new(m.unwrap()),
                        function: Box::new(f),
                    });
                    let apply = ast::Expr::Apply(ast::Apply {
                        span,
                        callee: Box::new(remote),
                        args,
                    });
                    self.expr(apply)
                }
                ast::Expr::Literal(ast::Literal::Atom(f)) => {
                    let (args, pre) = self.safe_list(f.span(), args)?;
                    let op = IExpr::Var(Var {
                        annotations: Annotations::default(),
                        name: f,
                        arity: Some(args.len()),
                    });
                    let apply = IExpr::Apply(IApply {
                        span,
                        annotations: Annotations::default(),
                        callee: vec![op],
                        args,
                    });
                    Ok((apply, pre))
                }
                callee => {
                    let (fun, mut pre) = self.safe(callee)?;
                    let (args, mut pre2) = self.safe_list(span, args)?;
                    pre.append(&mut pre2);
                    let apply = IExpr::Apply(IApply {
                        span,
                        annotations: Annotations::default(),
                        callee: vec![fun],
                        args,
                    });
                    Ok((apply, pre))
                }
            },
            ast::Expr::Match(ast::Match {
                span,
                pattern: pattern0,
                expr: expr0,
            }) => {
                // First fold matches together to create aliases
                let (pattern1, expr1) = fold_match(*expr0, *pattern0);
                let prev_wanted = self.set_wanted(&pattern1);
                let (expr2, mut pre) = self.novars(expr1.clone())?;
                self.context_mut().set_wanted(prev_wanted);
                let pattern2 = self.pattern(pattern1.clone());
                let fpat = self.context_mut().next_var(Some(span));
                let fail = fail_clause(
                    span,
                    vec![IExpr::Var(fpat.clone())],
                    ituple!(span, iatom!(span, symbols::Badmatch), IExpr::Var(fpat)),
                );
                match pattern2 {
                    Err(_) => {
                        // The pattern will not match. We must take care here to
                        // bind all variables that the pattern would have bound
                        // so that subsequent expressions do not refer to unbound
                        // variables.
                        //
                        // As an example, this code:
                        //
                        //   [X] = {Y} = E,
                        //   X + Y.
                        //
                        // will be rewritten to:
                        //
                        //   error({badmatch,E}),
                        //   case E of
                        //      {[X],{Y}} ->
                        //        X + Y;
                        //      Other ->
                        //        error({badmatch,Other})
                        //   end.
                        //
                        self.diagnostics
                            .diagnostic(Severity::Warning)
                            .with_message("bad pattern")
                            .with_primary_label(span, "this pattern cannot match")
                            .emit();
                        let (expr, mut pre) = self.safe(expr1)?;
                        let sanpat = sanitize(pattern1);
                        let sanpat = self.pattern(sanpat)?;
                        let badmatch = ituple!(span, iatom!(span, symbols::Badmatch), expr.clone());
                        let fail2 =
                            IExpr::PrimOp(IPrimOp::new(span, symbols::MatchFail, vec![badmatch]));
                        pre.push(fail2);
                        let mexpr = IExpr::Match(IMatch {
                            span,
                            annotations: Annotations::default(),
                            pattern: Box::new(sanpat),
                            arg: Box::new(expr),
                            guards: vec![],
                            fail,
                        });
                        Ok((mexpr, pre))
                    }
                    Ok(pattern2) => {
                        // We must rewrite top-level aliases to lets to avoid unbound
                        // variables in code such as:
                        //
                        //     <<42:Sz>> = Sz = B
                        //
                        // If we would keep the top-level aliases the example would
                        // be translated like this:
                        //
                        //     case B of
                        //         <Sz = #{#<42>(Sz,1,'integer',['unsigned'|['big']])}#>
                        //            when 'true' ->
                        //            .
                        //            .
                        //            .
                        //
                        // Here the variable Sz would be unbound in the binary pattern.
                        //
                        // Instead we bind Sz in a let to ensure it is bound when
                        // used in the binary pattern:
                        //
                        //     let <Sz> = B
                        //     in case Sz of
                        //         <#{#<42>(Sz,1,'integer',['unsigned'|['big']])}#>
                        //            when 'true' ->
                        //            .
                        //            .
                        //            .
                        //
                        let (pattern3, expr3, mut pre2) = letify_aliases(pattern2, expr2);
                        pre.append(&mut pre2);
                        let mexpr = IExpr::Match(IMatch {
                            span,
                            annotations: Annotations::default(),
                            pattern: Box::new(pattern3),
                            arg: Box::new(expr3),
                            guards: vec![],
                            fail,
                        });
                        Ok((mexpr, pre))
                    }
                }
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr {
                op: BinaryOp::Append,
                lhs,
                rhs,
                span,
            }) if lhs.is_lc() => {
                // Optimise '++' here because of the list comprehension algorithm.
                //
                // To avoid achieving quadratic complexity if there is a chain of
                // list comprehensions without generators combined with '++', force
                // evaluation of More now. Evaluating More here could also reduce the
                // number variables in the environment for letrec.
                let (rhs, mut rpre) = self.safe(*rhs)?;
                let lc = (*lhs).to_lc();
                let expr = *lc.body;
                let qualifiers = self.preprocess_quals(lc.qualifiers)?;
                let (y, mut ypre) = self.lc_tq(span, expr, qualifiers, rhs)?;
                rpre.append(&mut ypre);
                Ok((y, rpre))
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr {
                op: BinaryOp::AndAlso,
                lhs,
                rhs,
                span,
            }) => {
                let v = self.context_mut().next_var(Some(span));
                let atom_false =
                    ast::Expr::Literal(ast::Literal::Atom(Ident::new(symbols::False, span)));
                let expr = make_bool_switch(span, *lhs, ast::Var(v.name), *rhs, atom_false);
                self.expr(expr)
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr {
                op: BinaryOp::OrElse,
                lhs,
                rhs,
                span,
            }) => {
                let v = self.context_mut().next_var(Some(span));
                let atom_true =
                    ast::Expr::Literal(ast::Literal::Atom(Ident::new(symbols::True, span)));
                let expr = make_bool_switch(span, *lhs, ast::Var(v.name), atom_true, *rhs);
                self.expr(expr)
            }
            ast::Expr::BinaryExpr(ast::BinaryExpr { op, lhs, rhs, span }) => {
                let (args, pre) = self.safe_list(span, vec![*lhs, *rhs])?;
                let call = IExpr::Call(ICall::new(span, symbols::Erlang, op.to_symbol(), args));
                Ok((call, pre))
            }
            ast::Expr::UnaryExpr(ast::UnaryExpr { op, operand, span }) => {
                let (operand, pre) = self.safe(*operand)?;
                let call = IExpr::Call(ICall::new(
                    span,
                    symbols::Erlang,
                    op.to_symbol(),
                    vec![operand],
                ));
                Ok((call, pre))
            }
            _ => unreachable!(),
        }
    }

    fn fun_tq(
        &mut self,
        span: SourceSpan,
        name: Option<Ident>,
        clauses: Vec<ast::Clause>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let arity = clauses[0].patterns.len();
        let clauses = self.clauses(clauses)?;
        let vars = (0..arity)
            .map(|_| self.context_mut().next_var(Some(span)))
            .collect::<Vec<_>>();
        let ps = (0..arity)
            .map(|_| IExpr::Var(self.context_mut().next_var(Some(span))))
            .collect::<Vec<_>>();
        let fail = fail_clause(
            span,
            ps,
            IExpr::Literal(lit_tuple!(span, lit_atom!(span, symbols::FunctionClause))),
        );
        Ok((
            IExpr::Fun(IFun {
                span,
                annotations: Annotations::default(),
                id: Some(Ident::new(self.context_mut().new_fun_name(None), span)),
                name,
                vars,
                clauses,
                fail,
            }),
            vec![],
        ))
    }

    /// This is the implementation of the TQ translation scheme as described in _The Implementation of Functional Programming Languages_,
    /// Simon Peyton Jones, et al. pp 127-138
    fn lc_tq(
        &mut self,
        span: SourceSpan,
        body: ast::Expr,
        mut qualifiers: Vec<IQualifier>,
        last: IExpr,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let mut qs = qualifiers.drain(..);
        match qs.next() {
            None => {
                let (h1, mut hps) = self.safe(body)?;
                let (t1, mut tps) = force_safe(self.context_mut(), last);
                hps.append(&mut tps);
                let annotations = Annotations::default_compiler_generated();
                let expr = IExpr::Cons(ICons {
                    span,
                    annotations,
                    head: Box::new(h1),
                    tail: Box::new(t1),
                });
                Ok((expr, hps))
            }
            Some(IQualifier::Filter(filter)) => {
                self.filter_tq(span, body, filter, last, qs.collect(), true)
            }
            Some(IQualifier::Generator(gen)) => {
                let name = self.context_mut().new_fun_name(Some("lc"));
                let f = Var::new_with_arity(Ident::new(name, span), 1);
                let tail = gen.tail.unwrap();
                let nc = IExpr::Apply(IApply::new(
                    span,
                    IExpr::Var(f.clone()),
                    vec![IExpr::Var(tail.clone())],
                ));
                let fcvar = self.context_mut().next_var(Some(span));
                let var = self.context_mut().next_var(Some(span));
                let fail = bad_generator(span, vec![IExpr::Var(fcvar.clone())], fcvar.clone());
                let tail_clause = IClause {
                    span,
                    annotations: Annotations::default(),
                    patterns: vec![*gen.tail_pattern],
                    guards: vec![],
                    body: vec![last],
                };
                let clauses = match (gen.acc_pattern, gen.skip_pattern) {
                    (None, None) => vec![tail_clause],
                    (None, Some(skip_pat)) => {
                        let skip_clause = IClause {
                            span,
                            annotations: Annotations::from([
                                symbols::SkipClause,
                                symbols::CompilerGenerated,
                            ]),
                            patterns: vec![*skip_pat],
                            guards: vec![],
                            body: vec![nc],
                        };
                        vec![skip_clause, tail_clause]
                    }
                    (Some(acc_pat), Some(skip_pat)) => {
                        let skip_clause = IClause {
                            span,
                            annotations: Annotations::from([
                                symbols::SkipClause,
                                symbols::CompilerGenerated,
                            ]),
                            patterns: vec![*skip_pat],
                            guards: vec![],
                            body: vec![nc.clone()],
                        };
                        let (lc, mut lps) = self.lc_tq(span, body, qs.collect(), nc)?;
                        lps.push(lc);
                        let acc_clause = IClause {
                            span,
                            annotations: Annotations::default(),
                            patterns: vec![*acc_pat],
                            guards: gen.acc_guards,
                            body: lps,
                        };
                        vec![acc_clause, skip_clause, tail_clause]
                    }
                    _ => unreachable!(),
                };
                let fun = IExpr::Fun(IFun {
                    span,
                    annotations: Annotations::default(),
                    id: Some(Ident::new(name, span)),
                    name: None,
                    vars: vec![var],
                    clauses,
                    fail,
                });
                let mut body = gen.pre;
                body.push(IExpr::Apply(IApply {
                    span,
                    annotations: Annotations::default(),
                    callee: vec![IExpr::Var(f.clone())],
                    args: vec![*gen.arg],
                }));
                let expr = IExpr::LetRec(ILetRec {
                    span,
                    annotations: Annotations::from([symbols::ListComprehension]),
                    defs: vec![(f, fun)],
                    body,
                });
                Ok((expr, vec![]))
            }
        }
    }

    // bc_tq(Line, Exp, [Qualifier], More, State) -> {LetRec,[PreExp],State}.
    //  This TQ from Gustafsson ERLANG'05.
    //  More could be transformed before calling bc_tq.
    fn bc_tq(
        &mut self,
        span: SourceSpan,
        body: ast::Expr,
        mut qualifiers: Vec<IQualifier>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let binvar = self.context_mut().next_var(Some(span));
        let mut pre = vec![];
        if let Some(IQualifier::Generator(ref mut gen)) = qualifiers.first_mut() {
            pre.append(&mut gen.pre);
        }
        let (expr, mut bcpre) = self.bc_tq1(span, body, qualifiers, IExpr::Var(binvar.clone()))?;
        let initial_size = IExpr::Literal(lit_int!(span, Int::Small(256)));
        let init = IExpr::PrimOp(IPrimOp::new(
            span,
            symbols::BitsInitWritable,
            vec![initial_size],
        ));
        pre.push(IExpr::Set(ISet::new(span, binvar, init)));
        pre.append(&mut bcpre);
        Ok((expr, pre))
    }

    fn bc_tq1(
        &mut self,
        span: SourceSpan,
        body: ast::Expr,
        mut qualifiers: Vec<IQualifier>,
        last: IExpr,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let mut qs = qualifiers.drain(..);
        match qs.next() {
            Some(IQualifier::Generator(gen)) => {
                let name = self.context_mut().new_fun_name(Some("lbc"));
                let vars = self.context_mut().next_n_vars(2, Some(span));
                let acc_var = vars[1].clone();
                let v1 = self.context_mut().next_var(Some(span));
                let v2 = self.context_mut().next_var(Some(span));
                let fcvars = vec![IExpr::Var(v1.clone()), IExpr::Var(v2)];
                let ignore = self.context_mut().next_var(Some(span));
                let f = Var::new_with_arity(Ident::new(name, span), 2);
                let fail = bad_generator(span, fcvars, v1);
                let tail_clause = IClause {
                    span,
                    annotations: Annotations::default(),
                    patterns: vec![*gen.tail_pattern, IExpr::Var(ignore.clone())],
                    guards: vec![],
                    body: vec![IExpr::Var(acc_var.clone())],
                };
                let clauses = match (gen.acc_pattern, gen.skip_pattern) {
                    (None, None) => vec![tail_clause],
                    (None, Some(skip_pat)) => {
                        let skip_clause = IClause {
                            span,
                            annotations: Annotations::from([
                                symbols::CompilerGenerated,
                                symbols::SkipClause,
                            ]),
                            patterns: vec![*skip_pat, IExpr::Var(ignore)],
                            guards: vec![],
                            body: vec![IExpr::Apply(IApply::new(
                                span,
                                IExpr::Var(f.clone()),
                                vec![IExpr::Var(gen.tail.unwrap())],
                            ))],
                        };
                        vec![skip_clause, tail_clause]
                    }
                    (Some(acc_pat), Some(skip_pat)) => {
                        let nc = IExpr::Apply(IApply::new(
                            span,
                            IExpr::Var(f.clone()),
                            vec![IExpr::Var(gen.tail.unwrap())],
                        ));
                        let skip_clause = IClause {
                            span,
                            annotations: Annotations::from([
                                symbols::CompilerGenerated,
                                symbols::SkipClause,
                            ]),
                            patterns: vec![*skip_pat, IExpr::Var(ignore.clone())],
                            guards: vec![],
                            body: vec![nc.clone()],
                        };
                        let (bc, mut body) =
                            self.bc_tq1(span, body, qs.collect(), IExpr::Var(acc_var.clone()))?;
                        body.push(IExpr::Set(ISet::new(span, acc_var, bc)));
                        body.push(nc);
                        let acc_clause = IClause {
                            span,
                            annotations: Annotations::default(),
                            patterns: vec![*acc_pat, IExpr::Var(ignore)],
                            guards: gen.acc_guards,
                            body,
                        };
                        vec![acc_clause, skip_clause, tail_clause]
                    }
                    _ => unreachable!(),
                };
                let fun = IExpr::Fun(IFun {
                    span,
                    annotations: Annotations::default(),
                    id: Some(Ident::new(name, span)),
                    name: Some(Ident::new(name, span)),
                    vars,
                    clauses,
                    fail,
                });
                // Inlining would disable the size calculation optimization for bs_init_writable
                let mut body = gen.pre;
                body.push(IExpr::Apply(IApply::new(
                    span,
                    IExpr::Var(f.clone()),
                    vec![*gen.arg, last],
                )));
                let expr = IExpr::LetRec(ILetRec {
                    span,
                    annotations: Annotations::from([symbols::ListComprehension, symbols::NoInline]),
                    defs: vec![(f, fun)],
                    body,
                });
                Ok((expr, vec![]))
            }
            Some(IQualifier::Filter(filter)) => {
                self.filter_tq(span, body, filter, last, qs.collect(), false)
            }
            None => {
                match body {
                    ast::Expr::Binary(bin) => {
                        self.bc_tq_build(bin.span(), vec![], last, bin.elements)
                    }
                    body => {
                        let span = body.span();
                        let specifier = Some(BinaryEntrySpecifier::Binary { unit: 1 });
                        let (expr, pre) = self.safe(body)?;
                        match expr {
                            IExpr::Var(v) => {
                                let var = ast::Expr::Var(ast::Var(Ident::new(v.name(), v.span())));
                                let els = vec![ast::BinaryElement {
                                    span,
                                    bit_expr: var,
                                    bit_size: None,
                                    specifier,
                                }];
                                self.bc_tq_build(span, pre, last, els)
                            }
                            IExpr::Literal(Literal {
                                span: lspan,
                                value: Lit::Binary(bitvec),
                                ..
                            }) => {
                                let els = vec![ast::BinaryElement {
                                    span,
                                    bit_expr: ast::Expr::Literal(ast::Literal::Binary(
                                        lspan, bitvec,
                                    )),
                                    bit_size: None,
                                    specifier,
                                }];
                                self.bc_tq_build(span, pre, last, els)
                            }
                            _expr => {
                                // Any other safe (cons, tuple, literal) is not a bitstring.
                                // Force the evaluation to fail and generate a warning
                                let els = vec![ast::BinaryElement {
                                    span,
                                    bit_expr: ast::Expr::Literal(ast::Literal::Atom(Ident::new(
                                        symbols::BadValue,
                                        span,
                                    ))),
                                    bit_size: None,
                                    specifier,
                                }];
                                self.bc_tq_build(span, pre, last, els)
                            }
                        }
                    }
                }
            }
        }
    }

    fn bc_tq_build(
        &mut self,
        span: SourceSpan,
        mut pre: Vec<IExpr>,
        last: IExpr,
        mut elements: Vec<ast::BinaryElement>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        match last {
            IExpr::Var(Var { name, .. }) => {
                let specifier = Some(BinaryEntrySpecifier::Binary { unit: 1 });
                let element = ast::BinaryElement {
                    span: name.span,
                    bit_expr: ast::Expr::Var(ast::Var(name)),
                    bit_size: None,
                    specifier,
                };
                elements.insert(0, element);
                let (mut expr, mut pre1) =
                    self.expr(ast::Expr::Binary(ast::Binary { span, elements }))?;
                pre.append(&mut pre1);
                {
                    let annos = expr.annotations_mut();
                    annos.set(symbols::CompilerGenerated);
                    annos.set(symbols::SingleUse);
                }
                Ok((expr, pre))
            }
            other => panic!(
                "unexpected accumulator expression, expected var, got: {:?}",
                &other
            ),
        }
    }

    // filter_tq(Line, Expr, Filter, Mc, State, [Qualifier], TqFun) ->
    //     {Case,[PreExpr],State}.
    //  Transform an intermediate comprehension filter to its intermediate case
    //  representation.
    fn filter_tq(
        &mut self,
        span: SourceSpan,
        expr: ast::Expr,
        filter: IFilter,
        last: IExpr,
        qualifiers: Vec<IQualifier>,
        is_lc: bool,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let (lc, mut lps) = if is_lc {
            self.lc_tq(span, expr, qualifiers, last.clone())?
        } else {
            self.bc_tq1(span, expr, qualifiers, last.clone())?
        };
        lps.push(lc);
        let span = filter.span;
        match filter.filter {
            FilterType::Match(pre, arg) => {
                // The filter is an expression, it is compiled to a case of degree 1 with 3 clauses,
                // one accumulating, one skipping, and the final one throwing {case_clause, Value} where Value
                // is the result of the filter and is not a boolean
                let fail_pat = IExpr::Var(self.context_mut().next_var(Some(span)));
                let fail = fail_clause(
                    span,
                    vec![fail_pat.clone()],
                    ituple!(span, iatom!(span, symbols::BadFilter), fail_pat),
                );
                let expr = IExpr::Case(ICase {
                    span,
                    annotations: Annotations::from(symbols::ListComprehension),
                    args: vec![*arg],
                    clauses: vec![
                        IClause {
                            span,
                            annotations: Annotations::default(),
                            patterns: vec![iatom!(span, symbols::True)],
                            guards: vec![],
                            body: lps,
                        },
                        IClause {
                            span,
                            annotations: Annotations::default_compiler_generated(),
                            patterns: vec![iatom!(span, symbols::False)],
                            guards: vec![],
                            body: vec![last],
                        },
                    ],
                    fail,
                });
                Ok((expr, pre))
            }
            FilterType::Guard(guards) => {
                // The filter is a guard, compiled to an if, where if the guard succeeds,
                // then the comprehension continues, otherwise the current element is skipped
                let expr = IExpr::If(IIf {
                    span,
                    annotations: Annotations::from(symbols::ListComprehension),
                    guards,
                    then_body: lps,
                    else_body: vec![last],
                });
                Ok((expr, vec![]))
            }
        }
    }

    // preprocess_quals(Line, [Qualifier], State) -> {[Qualifier'],State}.
    //  Preprocess a list of Erlang qualifiers into its intermediate representation,
    //  represented as a list of IGen and IFilter records. We recognise guard
    //  tests and try to fold them together and join to a preceding generators, this
    //  should give us better and more compact code.
    fn preprocess_quals(
        &mut self,
        mut qualifiers: Vec<ast::Expr>,
    ) -> anyhow::Result<Vec<IQualifier>> {
        let mut acc = Vec::with_capacity(qualifiers.len());
        let mut iter = qualifiers.drain(..).peekable();
        while let Some(qualifier) = iter.next() {
            if qualifier.is_generator() {
                let mut guards = vec![];
                loop {
                    match iter.peek().map(is_guard_test) {
                        None | Some(false) => break,
                        Some(true) => guards.push(iter.next().unwrap()),
                    }
                }
                match qualifier {
                    ast::Expr::Generator(gen) => {
                        acc.push(IQualifier::Generator(self.generator(gen, guards)?))
                    }
                    _ => unreachable!(),
                }
            } else if is_guard_test(&qualifier) {
                let qspan = qualifier.span();
                // When a filter is a guard test, its argument in the IFilter record
                // is a list as returned by lc_guard_tests/2
                let mut guards = vec![];
                loop {
                    match iter.peek().map(is_guard_test) {
                        None | Some(false) => break,
                        Some(true) => guards.push(iter.next().unwrap()),
                    }
                }
                guards.insert(0, qualifier);
                let cg = self.lc_guard_tests(qspan, guards);
                acc.push(IQualifier::Filter(IFilter::new_guard(qspan, cg)))
            } else {
                let qspan = qualifier.span();
                // Otherwise, it is a pair {Pre, Arg} as in a generator input
                let (expr, pre) = self.novars(qualifier)?;
                acc.push(IQualifier::Filter(IFilter::new_match(qspan, pre, expr)))
            }
        }
        Ok(acc)
    }

    /// generator(Line, Generator, Guard, State) -> {Generator',State}.
    ///  Transform a given generator into its #igen{} representation.
    fn generator(&mut self, gen: ast::Generator, guards: Vec<ast::Expr>) -> anyhow::Result<IGen> {
        // Generators are abstracted as sextuplets:
        //  - acc_pat is the accumulator pattern, e.g. [Pat|Tail] for Pat <- Expr.
        //  - acc_guard is the list of guards immediately following the current
        //    generator in the qualifier list input.
        //  - skip_pat is the skip pattern, e.g. <<X,_:X,Tail/bitstring>> for
        //    <<X,1:X>> <= Expr.
        //  - tail is the variable used in AccPat and SkipPat bound to the rest of the
        //    generator input.
        //  - tail_pat is the tail pattern, respectively [] and <<_/bitstring>> for list
        //    and bit string generators.
        //  - arg is a pair {Pre,Arg} where Pre is the list of expressions to be
        //    inserted before the comprehension function and Arg is the expression
        //    that it should be passed.
        //
        match gen.ty {
            ast::GeneratorType::Default => {
                self.list_generator(gen.span, *gen.pattern, *gen.expr, guards)
            }
            ast::GeneratorType::Bitstring => {
                self.bit_generator(gen.span, *gen.pattern, *gen.expr, guards)
            }
        }
    }

    fn list_generator(
        &mut self,
        span: SourceSpan,
        pattern: ast::Expr,
        expr: ast::Expr,
        guards: Vec<ast::Expr>,
    ) -> anyhow::Result<IGen> {
        let head = self.pattern(pattern).ok();
        let tail = self.context_mut().next_var(Some(span));
        let skip = IExpr::Var(self.context_mut().next_var(Some(span)));
        let acc_guards = self.lc_guard_tests(span, guards);
        let (acc_pattern, skip_pattern) = match head {
            Some(head @ IExpr::Var(_)) => {
                // If the generator pattern is a variable, the pattern
                // from the accumulator clause can be reused in the skip one.
                // lc_tq and gc_tq1 takes care of dismissing the latter in that case.
                let cons = Box::new(IExpr::Cons(ICons::new(
                    span,
                    head,
                    IExpr::Var(tail.clone()),
                )));
                (Some(cons.clone()), Some(cons))
            }
            Some(head) => {
                let acc = Box::new(IExpr::Cons(ICons::new(
                    span,
                    head,
                    IExpr::Var(tail.clone()),
                )));
                let skip = Box::new(IExpr::Cons(ICons::new(
                    span,
                    skip.clone(),
                    IExpr::Var(tail.clone()),
                )));
                (Some(acc), Some(skip))
            }
            None => {
                // If it never matches, there is no need for an accumulator clause.
                (
                    None,
                    Some(Box::new(IExpr::Cons(ICons::new(
                        span,
                        skip.clone(),
                        IExpr::Var(tail.clone()),
                    )))),
                )
            }
        };
        let (arg, pre) = self.safe(expr)?;
        Ok(IGen {
            span,
            annotations: Annotations::default(),
            acc_pattern,
            acc_guards,
            skip_pattern,
            tail: Some(tail),
            tail_pattern: Box::new(IExpr::Literal(Literal::nil(span))),
            pre,
            arg: Box::new(arg),
        })
    }

    fn bit_generator(
        &mut self,
        span: SourceSpan,
        pattern: ast::Expr,
        expr: ast::Expr,
        guards: Vec<ast::Expr>,
    ) -> anyhow::Result<IGen> {
        match self.pattern(pattern)? {
            IExpr::Binary(IBinary {
                span,
                annotations,
                segments,
            }) => {
                // The function append_tail_segment/2 keeps variable patterns as-is, making
                // it possible to have the same skip clause removal as with list generators.
                let (acc_segments, tail, tail_segment) = self.append_tail_segment(span, segments);
                let acc_pattern = Box::new(IExpr::Binary(IBinary {
                    span,
                    annotations: annotations.clone(),
                    segments: acc_segments.clone(),
                }));
                let acc_guards = self.lc_guard_tests(span, guards);
                let skip_segments = self.skip_segments(acc_segments);
                let skip_pattern = Box::new(IExpr::Binary(IBinary {
                    span,
                    annotations,
                    segments: skip_segments,
                }));
                let (arg, pre) = self.safe(expr)?;
                Ok(IGen {
                    span,
                    annotations: Annotations::default(),
                    acc_pattern: Some(acc_pattern),
                    acc_guards,
                    skip_pattern: Some(skip_pattern),
                    tail: Some(tail),
                    tail_pattern: Box::new(IExpr::Binary(IBinary {
                        span,
                        annotations: Annotations::default(),
                        segments: vec![tail_segment],
                    })),
                    pre,
                    arg: Box::new(arg),
                })
            }
            _ => {
                // nomatch
                let (arg, pre) = self.safe(expr)?;
                Ok(IGen {
                    span,
                    annotations: Annotations::default(),
                    acc_pattern: None,
                    acc_guards: vec![],
                    skip_pattern: None,
                    tail: None,
                    tail_pattern: Box::new(IExpr::Var(Var::new(Ident::new(
                        symbols::Underscore,
                        span,
                    )))),
                    pre,
                    arg: Box::new(arg),
                })
            }
        }
    }

    fn lc_guard_tests(&mut self, span: SourceSpan, guards: Vec<ast::Expr>) -> Vec<IExpr> {
        if guards.is_empty() {
            return vec![];
        }
        let guards = self.guard_tests(span, guards);
        self.context_mut().in_guard = true;
        let guards = self.gexpr_top(guards);
        self.context_mut().in_guard = false;
        guards
    }

    fn append_tail_segment(
        &mut self,
        span: SourceSpan,
        mut segments: Vec<IBitstring>,
    ) -> (Vec<IBitstring>, Var, IBitstring) {
        let var = self.context_mut().next_var(None);
        let tail = IBitstring {
            span,
            annotations: Annotations::default(),
            value: Box::new(IExpr::Var(var.clone())),
            size: vec![],
            spec: BinaryEntrySpecifier::Binary { unit: 1 },
        };
        segments.push(tail.clone());
        (segments, var, tail)
    }

    // skip_segments(Segments, St0, Acc) -> {SkipSegments,St}.
    //  Generate the segments for a binary pattern that can be used
    //  in the skip clause that will continue the iteration when
    //  the accumulator pattern didn't match.
    fn skip_segments(&mut self, mut segments: Vec<IBitstring>) -> Vec<IBitstring> {
        let mut out = Vec::new();
        for mut segment in segments.drain(..) {
            if segment.value.is_var() {
                // We must keep the names of existing variables to ensure that patterns such as `<<Size, X:Size>>` will work.
                out.push(segment);
            } else {
                // Replace literal or expression with a variable (whose value will be ignored)
                let var = self.context_mut().next_var(None);
                *segment.value.as_mut() = IExpr::Var(var);
            }
        }
        out
    }

    fn expr_map(
        &mut self,
        span: SourceSpan,
        map: ast::Expr,
        fields: Vec<ast::MapField>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let (map, mut pre) = self.safe_map(map)?;
        let badmap = self.badmap_term(&map);
        let fail = fail_body(span, badmap);
        // fail.annotate(symbols::EvalFailure, Annotation::Term(Literal::atom(symbols::Badmap)));
        let is_empty = fields.is_empty();
        let (map2, mut pre2) = self.map_build_pairs(span, map.clone(), fields)?;
        pre.append(&mut pre2);
        let map3 = if is_empty { map.clone() } else { map2.clone() };
        let is_map = IExpr::Call(ICall::new(span, symbols::Erlang, symbols::IsMap, vec![map]));
        let case = IExpr::If(IIf {
            span,
            annotations: Annotations::default_compiler_generated(),
            guards: vec![is_map],
            then_body: vec![map3],
            else_body: vec![fail],
        });
        Ok((case, pre))
    }

    // expr_bin([ArgExpr], St) -> {[Arg],[PreExpr],St}.
    //  Flatten the arguments of a bin. Do this straight left to right!
    //  Note that ibinary needs to have its annotation wrapped in a #a{}
    //  record whereas c_literal should not have a wrapped annotation
    fn expr_bin(
        &mut self,
        span: SourceSpan,
        mut elements: Vec<ast::BinaryElement>,
    ) -> Result<(IExpr, Vec<IExpr>), Vec<IExpr>> {
        self.bin_elements(span, elements.as_mut_slice(), 1)
            .map_err(|_| vec![])?;
        match self.constant_bin(elements.as_slice()) {
            Ok(bin) => Ok((IExpr::Literal(Literal::binary(span, bin)), vec![])),
            Err(_) => {
                let (elements, pre) = self.expr_bin_1(elements)?;
                if elements.is_empty() {
                    Ok((IExpr::Literal(Literal::binary(span, BitVec::new())), vec![]))
                } else {
                    Ok((
                        IExpr::Binary(IBinary {
                            span,
                            annotations: Annotations::default(),
                            segments: elements,
                        }),
                        pre,
                    ))
                }
            }
        }
    }

    fn bin_elements(
        &mut self,
        span: SourceSpan,
        elements: &mut [ast::BinaryElement],
        _segment_id: usize,
    ) -> Result<(), ()> {
        let mut failed = false;
        for element in elements.iter_mut() {
            match make_bit_type(
                element.span,
                element.bit_size.as_ref(),
                element.specifier.clone(),
            ) {
                Ok((size, spec)) => {
                    element.bit_size = size;
                    element.specifier.replace(spec);
                }
                Err(reason) => {
                    failed = true;
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid binary element")
                        .with_primary_label(element.span, reason)
                        .with_secondary_label(span, "in this binary expression")
                        .emit();
                }
            }
        }
        if failed {
            Err(())
        } else {
            Ok(())
        }
    }

    fn expr_bin_1(
        &mut self,
        mut elements: Vec<ast::BinaryElement>,
    ) -> Result<(Vec<IBitstring>, Vec<IExpr>), Vec<IExpr>> {
        let mut segments = Vec::with_capacity(elements.len());
        let mut pre = vec![];
        let mut is_bad = false;
        for element in elements.drain(..) {
            match self.bitstr(element) {
                Ok((mut segments2, mut pre2)) if !is_bad => {
                    segments.append(&mut segments2);
                    pre.append(&mut pre2);
                }
                Ok((_segments2, mut pre2)) => {
                    pre.append(&mut pre2);
                }
                Err(mut pre2) => {
                    is_bad = true;
                    pre.append(&mut pre2);
                }
            }
        }

        if is_bad {
            Err(pre)
        } else {
            Ok((segments, pre))
        }
    }

    fn constant_bin(&mut self, elements: &[ast::BinaryElement]) -> Result<BitVec, ()> {
        verify_suitable_fields(elements)?;
        let mut bindings = evaluator::Bindings::default();
        evaluator::expr_grp(elements, &mut bindings, |expr, _bindings| match expr {
            ast::Expr::Literal(ast::Literal::Atom(_))
            | ast::Expr::Literal(ast::Literal::String(_))
            | ast::Expr::Literal(ast::Literal::Char(_, _))
            | ast::Expr::Literal(ast::Literal::Integer(_, _))
            | ast::Expr::Literal(ast::Literal::Float(_, _)) => Ok(expr),
            _ => Err(()),
        })
    }

    fn bitstrs(
        &mut self,
        mut elements: Vec<ast::BinaryElement>,
    ) -> Result<(Vec<IBitstring>, Vec<IExpr>), Vec<IExpr>> {
        let mut segments = Vec::with_capacity(elements.len());
        let mut pre = vec![];
        for element in elements.drain(..) {
            let (mut seg, mut pre2) = self.bitstr(element)?;
            segments.append(&mut seg);
            pre.append(&mut pre2);
        }
        Ok((segments, pre))
    }

    fn bitstr(
        &mut self,
        element: ast::BinaryElement,
    ) -> Result<(Vec<IBitstring>, Vec<IExpr>), Vec<IExpr>> {
        use firefly_binary::BinaryEntrySpecifier as S;

        let span = element.span;
        let size_opt = element.bit_size;
        let spec = element.specifier;
        match element.bit_expr {
            ast::Expr::Literal(ast::Literal::String(s)) => {
                if matches!(size_opt, Some(ast::Expr::Literal(ast::Literal::Integer(_, ref i))) if i == &8)
                {
                    // NOTE(pauls): I have no idea why erlc special cases this, but
                    // it ignores the spec entirely and hashes the string as an integer
                    // value
                    //
                    // <<"foobar":8>>
                    self.bitstrs(bin_expand_string(s, 0i64.into(), 0))
                } else if s.name == symbols::Empty {
                    // Empty string. We must make sure that the type is correct.
                    let (mut bs, mut pre) = self.bitstr(ast::BinaryElement {
                        span,
                        bit_expr: ast::Expr::Literal(ast::Literal::Integer(span, 0.into())),
                        bit_size: size_opt.clone(),
                        specifier: spec,
                    })?;
                    let bs = bs.pop().unwrap();
                    // At this point, the type is either a correct literal or an expression
                    assert!(bs.size.len() < 2);
                    match bs.size.first() {
                        None => {
                            // One of the utf* types. The size is not used.
                            debug_assert!(!spec.unwrap_or_default().has_size());
                            Ok((vec![], vec![]))
                        }
                        Some(IExpr::Literal(Literal {
                            value: Lit::Integer(i),
                            ..
                        })) if i >= &0 => Ok((vec![], vec![])),
                        Some(IExpr::Var(_)) => {
                            // Must add a test to verify that the size expression is an integer >= 0
                            let size = size_opt.unwrap();
                            let test0 = ast::Expr::Apply(ast::Apply::remote(
                                span,
                                symbols::Erlang,
                                symbols::IsInteger,
                                vec![size.clone()],
                            ));
                            let test1 = ast::Expr::Apply(ast::Apply::remote(
                                span,
                                symbols::Erlang,
                                symbols::Gte,
                                vec![
                                    size.clone(),
                                    ast::Expr::Literal(ast::Literal::Integer(span, 0.into())),
                                ],
                            ));
                            let test2 = ast::Expr::BinaryExpr(ast::BinaryExpr::new(
                                span,
                                BinaryOp::AndAlso,
                                test0,
                                test1,
                            ));
                            let fail = ast::Expr::Apply(ast::Apply::remote(
                                span,
                                symbols::Erlang,
                                symbols::Error,
                                vec![ast::Expr::Literal(ast::Literal::Atom(Ident::new(
                                    symbols::Badarg,
                                    span,
                                )))],
                            ));
                            let test = ast::Expr::BinaryExpr(ast::BinaryExpr::new(
                                span,
                                BinaryOp::OrElse,
                                test2,
                                fail,
                            ));
                            let mexpr = ast::Expr::Match(ast::Match {
                                span,
                                pattern: Box::new(ast::Expr::Var(ast::Var(Ident::new(
                                    symbols::Underscore,
                                    span,
                                )))),
                                expr: Box::new(test),
                            });
                            let (_, mut pre2) = self.expr(mexpr).unwrap();
                            pre.append(&mut pre2);
                            Ok((vec![], pre))
                        }
                        Some(other) => panic!("invalid bitstring expression: {:?}", &other),
                    }
                } else {
                    let (mut bs, pre) = self.bitstr(ast::BinaryElement {
                        span,
                        bit_expr: ast::Expr::Literal(ast::Literal::Integer(span, 0.into())),
                        bit_size: size_opt,
                        specifier: spec,
                    })?;
                    let bitstr = bs.pop().unwrap();
                    let segments = s
                        .as_str()
                        .get()
                        .chars()
                        .map(|c| {
                            let mut b = bitstr.clone();
                            *b.value.as_mut() = iint!(span, c as i64);
                            b
                        })
                        .collect();
                    Ok((segments, pre))
                }
            }
            expr => {
                let spec = spec.unwrap_or_default();
                let (expr, mut pre) = self.safe(expr).map_err(|_| vec![])?;
                let (size, mut pre2) = match self.safe(size_opt.unwrap()) {
                    Ok(ok) => ok,
                    Err(_) => return Err(pre),
                };
                pre.append(&mut pre2);

                match (spec, &expr) {
                    (_, IExpr::Var(_))
                    | (
                        S::Integer { .. },
                        IExpr::Literal(Literal {
                            value: Lit::Integer(_),
                            ..
                        }),
                    )
                    | (
                        S::Utf8 { .. },
                        IExpr::Literal(Literal {
                            value: Lit::Integer(_),
                            ..
                        }),
                    )
                    | (
                        S::Utf16 { .. },
                        IExpr::Literal(Literal {
                            value: Lit::Integer(_),
                            ..
                        }),
                    )
                    | (
                        S::Utf32 { .. },
                        IExpr::Literal(Literal {
                            value: Lit::Integer(_),
                            ..
                        }),
                    ) => (),
                    (S::Float { .. }, IExpr::Literal(Literal { value: lit, .. }))
                        if lit.is_number() =>
                    {
                        ()
                    }
                    (
                        S::Binary { .. },
                        IExpr::Literal(Literal {
                            value: Lit::Binary(_),
                            ..
                        }),
                    ) => (),
                    (_, _) => {
                        // Note that the pre expressions may bind variables that
                        // are used later or have side effects.
                        return Err(pre);
                    }
                }
                let size = match size {
                    v @ IExpr::Var(_) => vec![v],
                    IExpr::Literal(lit) => {
                        if lit.as_integer().map(|i| i >= &0).unwrap_or_default() {
                            vec![IExpr::Literal(lit)]
                        } else if let Some(atom) = lit.as_atom() {
                            match atom {
                                symbols::Undefined => vec![],
                                symbols::All => vec![IExpr::Literal(lit)],
                                _ => return Err(pre),
                            }
                        } else {
                            return Err(pre);
                        }
                    }
                    _ => return Err(pre),
                };
                // We will add a 'segment' annotation to segments that could
                // fail. There is no need to add it to literal segments of fixed
                // sized. The annotation will be used by the runtime system to
                // provide extended error information if construction of the
                // binary fails.
                // let anno = Annotations::from(vec![(symbols::Segment, lit_tuple!(span, lit_int!(span, line), lit_int!(span, column)))])
                let bs = IBitstring {
                    span,
                    annotations: Annotations::default(),
                    value: Box::new(expr),
                    size,
                    spec,
                };
                Ok((vec![bs], pre))
            }
        }
    }

    fn map_build_pairs(
        &mut self,
        span: SourceSpan,
        map: IExpr,
        fields: Vec<ast::MapField>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let (pairs, pre) = self.map_build_pairs1(fields)?;
        let map = IExpr::Map(IMap {
            span,
            annotations: Annotations::default(),
            arg: Box::new(map),
            pairs,
            is_pattern: false,
        });
        Ok((map, pre))
    }

    fn map_build_pairs1(
        &mut self,
        mut fields: Vec<ast::MapField>,
    ) -> anyhow::Result<(Vec<IMapPair>, Vec<IExpr>)> {
        let mut used: HashSet<Literal> = HashSet::new();
        let mut pairs = Vec::with_capacity(fields.len());
        let mut pre = Vec::new();
        for field in fields.drain(..) {
            let (op, key0, value0) = match field {
                ast::MapField::Assoc { key, value, .. } => (MapOp::Assoc, key, value),
                ast::MapField::Exact { key, value, .. } => (MapOp::Exact, key, value),
            };
            let (key, mut pre0) = self.safe(key0)?;
            let (value, mut pre1) = self.safe(value0)?;
            pre.append(&mut pre0);
            pre.append(&mut pre1);
            if let IExpr::Literal(ref lit) = &key {
                if let Some(prev) = used.get(lit) {
                    self.diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("duplicate map key")
                        .with_primary_label(lit.span, "this map key is repeated")
                        .with_secondary_label(prev.span, "it is was first used here")
                        .emit();
                } else {
                    used.insert(lit.clone());
                }
            }
            pairs.push(IMapPair {
                op,
                key: vec![key],
                value: Box::new(value),
            });
        }

        Ok((pairs, pre))
    }

    fn badmap_term(&self, map: &IExpr) -> IExpr {
        let span = map.span();
        let badmap = iatom!(span, symbols::Badmap);
        if self.context().in_guard {
            badmap
        } else {
            ituple!(span, badmap, map.clone())
        }
    }

    fn safe_map(&mut self, map: ast::Expr) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        match self.safe(map)? {
            ok @ (IExpr::Var(_), _) => Ok(ok),
            ok @ (
                IExpr::Literal(Literal {
                    value: Lit::Map(_), ..
                }),
                _,
            ) => Ok(ok),
            (notmap, mut pre) => {
                // Not a map. There will be a syntax error if we try to pretty-print
                // the Core Erlang code and then try to parse it. To avoid the syntax
                // error, force the term into a variable.
                let span = notmap.span();
                let v = self.context_mut().next_var(Some(span));
                pre.push(IExpr::Set(ISet {
                    span,
                    annotations: Annotations::default(),
                    var: v.clone(),
                    arg: Box::new(notmap),
                }));
                Ok((IExpr::Var(v), pre))
            }
        }
    }

    fn novars(&mut self, expr: ast::Expr) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let (expr, mut pre) = self.expr(expr)?;
        let (sexpr, mut pre2) = self.force_novars(expr);
        pre.append(&mut pre2);
        Ok((sexpr, pre))
    }

    fn force_novars(&mut self, expr: IExpr) -> (IExpr, Vec<IExpr>) {
        match expr {
            app @ IExpr::Apply(_) => (app, vec![]),
            call @ IExpr::Call(_) => (call, vec![]),
            fun @ IExpr::Fun(_) => (fun, vec![]),
            bin @ IExpr::Binary(_) => (bin, vec![]),
            map @ IExpr::Map(_) => (map, vec![]),
            other => force_safe(self.context_mut(), other),
        }
    }

    /// safe_list(Expr, State) -> {Safe,[PreExpr],State}.
    ///  Generate an internal safe expression for a list of
    ///  expressions.
    fn safe_list(
        &mut self,
        span: SourceSpan,
        mut exprs: Vec<ast::Expr>,
    ) -> anyhow::Result<(Vec<IExpr>, Vec<IExpr>)> {
        let mut out = Vec::<IExpr>::with_capacity(exprs.len());
        let mut pre = Vec::<Vec<IExpr>>::new();
        for expr in exprs.drain(..) {
            let (cexpr, pre2) = self.safe(expr)?;
            match pre.pop() {
                Some(mut prev) if prev.len() == 1 => {
                    match prev.pop().unwrap() {
                        IExpr::Exprs(IExprs { mut bodies, .. }) => {
                            // A cons within a cons
                            out.push(cexpr);
                            // [Pre2 | Bodies] ++ Pre
                            pre.extend(bodies.drain(..));
                            pre.push(pre2);
                        }
                        prev_expr => {
                            out.push(cexpr);
                            prev.push(prev_expr);
                            pre.push(prev);
                            pre.push(pre2);
                        }
                    }
                }
                Some(prev) => {
                    out.push(cexpr);
                    pre.push(prev);
                    pre.push(pre2);
                }
                None => {
                    out.push(cexpr);
                    pre.push(pre2);
                }
            }
        }

        let mut pre = pre
            .drain(..)
            .filter(|exprs| !exprs.is_empty())
            .collect::<Vec<_>>();
        match pre.len() {
            0 => Ok((out, vec![])),
            1 => Ok((out, pre.pop().unwrap())),
            _ => Ok((out, vec![IExpr::Exprs(IExprs::new(span, pre))])),
        }
    }

    // safe(Expr, State) -> {Safe,[PreExpr],State}.
    //  Generate an internal safe expression.  These are simples without
    //  binaries which can fail.  At this level we do not need to do a
    //  deep check.  Must do special things with matches here.
    fn safe(&mut self, expr: ast::Expr) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        let (expr, mut pre) = self.expr(expr)?;
        let (expr, mut pre2) = force_safe(self.context_mut(), expr);
        pre.append(&mut pre2);
        Ok((expr, pre))
    }

    // try_exception([ExcpClause]) -> {[ExcpVar],Handler}
    fn try_exception(&mut self, clauses: Vec<ast::Clause>) -> anyhow::Result<(Vec<Var>, IExpr)> {
        // Note that the tag is not needed for rethrow - it is already in the exception info
        let (tag, value, info) = {
            let context = self.context_mut();
            let tag = context.next_var(None);
            let value = context.next_var(None);
            let info = context.next_var(None);
            (tag, value, info)
        };
        let clauses = self.clauses(clauses)?;
        let clauses = try_build_stacktrace(clauses, tag.name);
        let span = clauses.get(0).map(|c| c.span).unwrap_or_default();
        let evars = vec![
            IExpr::Var(tag.clone()),
            IExpr::Var(value.clone()),
            IExpr::Var(info.clone()),
        ];
        let fail = Box::new(IClause {
            span,
            annotations: Annotations::default_compiler_generated(),
            patterns: evars.clone(),
            guards: vec![],
            body: vec![IExpr::PrimOp(IPrimOp::new(
                span,
                symbols::Raise,
                vec![
                    IExpr::Var(tag.clone()),
                    IExpr::Var(value.clone()),
                    IExpr::Var(info.clone()),
                ],
            ))],
        });
        let handler = IExpr::Case(ICase {
            span,
            annotations: Annotations::default(),
            args: evars,
            clauses,
            fail,
        });
        Ok((vec![tag, value, info], handler))
    }

    fn try_after(
        &mut self,
        span: SourceSpan,
        exprs: Vec<ast::Expr>,
        mut after: Vec<ast::Expr>,
    ) -> anyhow::Result<(IExpr, Vec<IExpr>)> {
        ta_sanitize_as(&mut after);
        let exprs = self.exprs(exprs)?;
        let after = self.exprs(after)?;
        let v = self.context_mut().next_var(Some(span));
        if is_iexprs_small(&after, 20) {
            Ok(self.try_after_small(span, exprs, after, v))
        } else {
            Ok(self.try_after_large(span, exprs, after, v))
        }
    }

    fn try_after_large(
        &mut self,
        span: SourceSpan,
        exprs: Vec<IExpr>,
        after: Vec<IExpr>,
        var: Var,
    ) -> (IExpr, Vec<IExpr>) {
        // Large 'after' block; break it out into a wrapper function to reduce code size
        let name = self.context_mut().new_fun_name(Some("after"));
        let fail = fail_clause(
            span,
            vec![],
            IExpr::Literal(lit_tuple!(span, lit_atom!(span, symbols::FunctionClause))),
        );
        let fun = IExpr::Fun(IFun {
            span,
            annotations: Annotations::default(),
            id: Some(Ident::new(name, span)),
            name: Some(Ident::new(name, span)),
            vars: vec![],
            clauses: vec![IClause::new(span, vec![], vec![], after)],
            fail,
        });
        let apply = IExpr::Apply(IApply {
            span,
            annotations: Annotations::default_compiler_generated(),
            callee: vec![IExpr::Var(Var {
                annotations: Annotations::default(),
                name: Ident::new(name, span),
                arity: Some(0),
            })],
            args: vec![],
        });
        let (evars, handler) = self.after_block(span, vec![apply.clone()]);
        let texpr = IExpr::Try(ITry {
            span,
            annotations: Annotations::default(),
            args: exprs,
            vars: vec![var.clone()],
            body: vec![apply, IExpr::Var(var)],
            evars,
            handler: Box::new(handler),
        });
        let letr = IExpr::LetRec(ILetRec {
            span,
            annotations: Annotations::default(),
            defs: vec![(Var::new_with_arity(Ident::new(name, span), 0), fun)],
            body: vec![texpr],
        });
        (letr, vec![])
    }

    fn try_after_small(
        &mut self,
        span: SourceSpan,
        exprs: Vec<IExpr>,
        mut after: Vec<IExpr>,
        var: Var,
    ) -> (IExpr, Vec<IExpr>) {
        // Small 'after' block; inline it
        let (evars, handler) = self.after_block(span, after.clone());
        after.push(IExpr::Var(var.clone()));
        let texpr = IExpr::Try(ITry {
            span,
            annotations: Annotations::default(),
            args: exprs,
            vars: vec![var],
            body: after,
            evars,
            handler: Box::new(handler),
        });
        (texpr, vec![])
    }

    fn after_block(&mut self, span: SourceSpan, mut after: Vec<IExpr>) -> (Vec<Var>, IExpr) {
        let (tag, value, info) = {
            let context = self.context_mut();
            let tag = context.next_var(Some(span));
            let value = context.next_var(Some(span));
            let info = context.next_var(Some(span));
            (tag, value, info)
        };
        let evs = vec![
            IExpr::Var(tag.clone()),
            IExpr::Var(value.clone()),
            IExpr::Var(info.clone()),
        ];
        after.push(IExpr::PrimOp(IPrimOp::new(
            span,
            symbols::Raise,
            vec![
                IExpr::Var(tag.clone()),
                IExpr::Var(info.clone()),
                IExpr::Var(value.clone()),
            ],
        )));
        let handler = IExpr::Case(ICase {
            span,
            annotations: Annotations::default(),
            args: evs.clone(),
            clauses: vec![],
            fail: Box::new(IClause {
                span,
                annotations: Annotations::default_compiler_generated(),
                patterns: evs,
                guards: vec![],
                body: after,
            }),
        });
        (vec![tag, value, info], handler)
    }

    fn set_wanted(&mut self, expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::Var(ast::Var(id)) => {
                if id.name == symbols::Empty {
                    self.context_mut().set_wanted(false)
                } else {
                    if id.name.as_str().get().starts_with('_') {
                        self.context_mut().set_wanted(false)
                    } else {
                        self.context().wanted
                    }
                }
            }
            _ => self.context().wanted,
        }
    }
}

fn fail_body(span: SourceSpan, arg: IExpr) -> IExpr {
    IExpr::PrimOp(IPrimOp::new(span, symbols::MatchFail, vec![arg]))
}

fn fail_clause(span: SourceSpan, patterns: Vec<IExpr>, arg: IExpr) -> Box<IClause> {
    Box::new(IClause {
        span,
        annotations: Annotations::default_compiler_generated(),
        patterns,
        guards: vec![],
        body: vec![fail_body(span, arg)],
    })
}

fn bad_generator(span: SourceSpan, patterns: Vec<IExpr>, generator: Var) -> Box<IClause> {
    let tuple = ituple!(
        span,
        IExpr::Literal(lit_atom!(span, symbols::BadGenerator)),
        IExpr::Var(generator)
    );
    let call = IExpr::PrimOp(IPrimOp::new(span, symbols::Error, vec![tuple]));
    Box::new(IClause {
        span,
        annotations: Annotations::default_compiler_generated(),
        patterns,
        guards: vec![],
        body: vec![call],
    })
}

/// sanitize(Pat) -> SanitizedPattern
///
/// Rewrite Pat so that it will be accepted by pattern/2 and will
/// bind the same variables as the original pattern.
///
/// Here is an example of a pattern that would cause a pattern/2
/// to generate a 'nomatch' exception:
///
/// ```erlang
///      #{k:=X,k:=Y} = [Z]
/// ```
///
///  The sanitized pattern will look like:
///
/// ```erlang
///      {{X,Y},[Z]}
/// ```
fn sanitize(expr: ast::Expr) -> ast::Expr {
    match expr {
        ast::Expr::Match(ast::Match {
            span,
            pattern,
            expr,
        }) => {
            let pattern = sanitize(*pattern);
            let expr = sanitize(*expr);
            ast::Expr::Tuple(ast::Tuple {
                span,
                elements: vec![pattern, expr],
            })
        }
        ast::Expr::Cons(ast::Cons { span, head, tail }) => {
            let head = Box::new(sanitize(*head));
            let tail = Box::new(sanitize(*tail));
            ast::Expr::Cons(ast::Cons { span, head, tail })
        }
        ast::Expr::Tuple(ast::Tuple { span, mut elements }) => {
            let elements = elements.drain(..).map(sanitize).collect();
            ast::Expr::Tuple(ast::Tuple { span, elements })
        }
        ast::Expr::Binary(ast::Binary {
            span,
            elements: mut bin_elements,
        }) => {
            let mut elements = vec![];
            for element in bin_elements.drain(..) {
                match element.bit_expr {
                    var @ ast::Expr::Var(_) => elements.push(var),
                    _ => (),
                }
            }
            ast::Expr::Tuple(ast::Tuple { span, elements })
        }
        ast::Expr::Map(ast::Map { span, mut fields }) => {
            let elements = fields
                .drain(..)
                .map(|field| match field {
                    ast::MapField::Exact { value, .. } => sanitize(value),
                    _ => unreachable!(),
                })
                .collect();
            ast::Expr::Tuple(ast::Tuple { span, elements })
        }
        ast::Expr::BinaryExpr(ast::BinaryExpr { span, lhs, rhs, .. }) => {
            let lhs = sanitize(*lhs);
            let rhs = sanitize(*rhs);
            ast::Expr::Tuple(ast::Tuple {
                span,
                elements: vec![lhs, rhs],
            })
        }
        expr => expr,
    }
}

/// unforce(Expr, PreExprList, BoolExprList) -> BoolExprList'.
///
/// Filter BoolExprList. BoolExprList is a list of simple expressions
/// (variables or literals) of which we are not sure whether they are booleans.
///
/// The basic idea for filtering is the following transformation:
///
/// ```erlang
///      (E =:= Bool) and is_boolean(E)   ==>  E =:= Bool
/// ```
///
/// where E is an arbitrary expression and Bool is 'true' or 'false'.
///
/// The transformation is still valid if there are other expressions joined
/// by 'and' operations:
///
/// ```erlang
///      E1 and (E2 =:= true) and E3 and is_boolean(E)   ==>  E1 and (E2 =:= true) and E3
/// ```
///
/// but expressions such as:
///
/// ```erlang
///     not (E =:= true) and is_boolean(E)
/// ```
///
/// or expression using 'or' or 'xor' cannot be transformed in this
/// way (such expressions are the reason for adding the is_boolean/1
/// test in the first place).
///
fn unforce(expr: &IExpr, mut pre: Vec<IExpr>, bools: Vec<IExpr>) -> Vec<IExpr> {
    if bools.is_empty() {
        bools
    } else {
        pre.push(expr.clone());
        let mut tree = BTreeMap::new();
        let expr = unforce_tree(pre, &mut tree);
        unforce2(&expr, bools)
    }
}

fn unforce2(expr: &IExpr, mut bools: Vec<IExpr>) -> Vec<IExpr> {
    match expr {
        IExpr::Call(call) if call.is_static(symbols::Erlang, symbols::And, 2) => {
            let bools = unforce2(call.args.get(0).unwrap(), bools);
            unforce2(call.args.get(1).unwrap(), bools)
        }
        IExpr::Call(call) if call.is_static(symbols::Erlang, symbols::EqualStrict, 2) => {
            if call.args[1].is_boolean() {
                let e = call.args.get(0).unwrap();
                match bools.iter().position(|b| b == e) {
                    None => bools,
                    Some(idx) => {
                        bools.remove(idx);
                        bools
                    }
                }
            } else {
                bools
            }
        }
        _ => bools,
    }
}

fn unforce_tree(mut exprs: Vec<IExpr>, tree: &mut BTreeMap<Symbol, IExpr>) -> IExpr {
    let mut it = exprs.drain(..);
    while let Some(expr) = it.next() {
        match expr {
            IExpr::Exprs(IExprs { mut bodies, .. }) => {
                let mut base = Vec::with_capacity(bodies.iter().fold(0, |acc, es| es.len() + acc));
                base.extend(bodies.drain(..).flat_map(|es| es));
                base.extend(it);
                return unforce_tree(base, tree);
            }
            IExpr::Set(ISet { var, arg, .. }) => {
                let arg = unforce_tree_subst(*arg, tree);
                tree.insert(var.name(), arg);
                continue;
            }
            call @ IExpr::Call(_) => {
                return unforce_tree_subst(call, tree);
            }
            IExpr::Var(var) => {
                return tree.remove(&var.name()).unwrap();
            }
            _ => unreachable!(),
        }
    }

    unreachable!()
}

fn unforce_tree_subst(expr: IExpr, tree: &mut BTreeMap<Symbol, IExpr>) -> IExpr {
    match expr {
        IExpr::Call(mut call) => {
            if call.is_static(symbols::Erlang, symbols::EqualStrict, 2) {
                if call.args[1].is_boolean() {
                    // We have erlang:'=:='(Expr, Bool). We must not expand this call any more
                    // or we will not recognize is_boolean(Expr) later.
                    return IExpr::Call(call);
                }
            }

            for arg in call.args.iter_mut() {
                if let IExpr::Var(v) = arg {
                    if let Some(value) = tree.get(&v.name()) {
                        *arg = value.clone();
                    }
                }
            }

            IExpr::Call(call)
        }
        expr => expr,
    }
}

fn letify_aliases(pattern: IExpr, expr: IExpr) -> (IExpr, IExpr, Vec<IExpr>) {
    match pattern {
        IExpr::Alias(IAlias {
            span, var, pattern, ..
        }) => {
            let (pattern, expr2, mut pre) = letify_aliases(*pattern, IExpr::Var(var.clone()));
            pre.insert(0, IExpr::Set(ISet::new(span, var, expr)));
            (pattern, expr2, pre)
        }
        pattern => (pattern, expr, vec![]),
    }
}

// Fold nested matches into one match with aliased patterns
fn fold_match(expr: ast::Expr, pattern: ast::Expr) -> (ast::Expr, ast::Expr) {
    match expr {
        ast::Expr::Match(ast::Match {
            span,
            expr: box expr0,
            pattern: box pattern0,
        }) => {
            let (pattern1, expr1) = fold_match(expr0, pattern);
            let pattern2 = ast::Expr::Match(ast::Match {
                span,
                pattern: Box::new(pattern0),
                expr: Box::new(pattern1),
            });
            (pattern2, expr1)
        }
        expr => (pattern, expr),
    }
}

fn make_bool_switch(
    span: SourceSpan,
    expr: ast::Expr,
    var: ast::Var,
    truep: ast::Expr,
    falsep: ast::Expr,
) -> ast::Expr {
    let badarg = ast::Expr::Literal(ast::Literal::Atom(Ident::new(symbols::Badarg, span)));
    let atom_true = ast::Expr::Literal(ast::Literal::Atom(Ident::new(symbols::True, span)));
    let atom_false = ast::Expr::Literal(ast::Literal::Atom(Ident::new(symbols::False, span)));
    let error = ast::Expr::Tuple(ast::Tuple {
        span,
        elements: vec![badarg, ast::Expr::Var(var)],
    });
    ast::Expr::Case(ast::Case {
        span,
        expr: Box::new(expr),
        clauses: vec![
            ast::Clause {
                span,
                patterns: vec![atom_true],
                guards: vec![],
                body: vec![truep],
                compiler_generated: true,
            },
            ast::Clause {
                span,
                patterns: vec![atom_false],
                guards: vec![],
                body: vec![falsep],
                compiler_generated: true,
            },
            ast::Clause {
                span,
                patterns: vec![ast::Expr::Var(var)],
                guards: vec![],
                body: vec![apply!(span, erlang, error, (error))],
                compiler_generated: true,
            },
        ],
    })
}

// 'after' blocks don't have a result, so we match the last expression with '_'
// to suppress false "unmatched return" warnings in tools that look at core
// Erlang, such as `dialyzer`.
fn ta_sanitize_as(exprs: &mut Vec<ast::Expr>) {
    let last = exprs.pop().unwrap();
    let span = last.span();
    exprs.push(ast::Expr::Match(ast::Match {
        span,
        pattern: Box::new(ast::Expr::Var(ast::Var(Ident::new(
            symbols::Underscore,
            span,
        )))),
        expr: Box::new(last),
    }));
}

fn try_build_stacktrace(mut clauses: Vec<IClause>, raw_stack: Ident) -> Vec<IClause> {
    let mut output = Vec::with_capacity(clauses.len());
    for mut clause in clauses.drain(..) {
        match clause.patterns.len() {
            1 => {
                // This occurs when the original input was Abstract Erlang format which treats
                // the clause pattern as an implicit tuple, so we need to unwrap the tuple before
                // proceeding
                match clause.patterns.pop().unwrap() {
                    IExpr::Tuple(ITuple { elements, .. }) => {
                        assert_eq!(
                            elements.len(),
                            3,
                            "unexpected number of catch clause pattern elements"
                        );
                        clause.patterns = elements;
                    }
                    other => panic!("unexpected catch clause pattern: {:#?}", &other),
                }
            }
            3 => (),
            n => panic!(
                "unexpected number of catch clause patterns, expected 1 or 3, got {}",
                n
            ),
        };
        let mut stk = clause.patterns.pop().unwrap();
        match stk {
            IExpr::Var(Var { name, .. }) if name == symbols::Underscore => {
                // Stacktrace variable is not used, nothing to do.
                stk.annotations_mut().set(symbols::RawStack);
                clause.patterns.push(stk);
                output.push(clause);
            }
            IExpr::Var(var) => {
                // Add code to build the stacktrace
                let span = var.span();
                let raw_stack = IExpr::Var(Var::new(raw_stack));
                clause.patterns.push(raw_stack.clone());
                let call = IExpr::PrimOp(IPrimOp::new(
                    span,
                    symbols::BuildStacktrace,
                    vec![raw_stack],
                ));
                let set = IExpr::Set(ISet {
                    span,
                    annotations: Annotations::default(),
                    var,
                    arg: Box::new(call),
                });
                clause.body.insert(0, set);
                output.push(clause);
            }
            _ => panic!("expected stacktrace variable"),
        }
    }

    output
}

// is_iexprs_small([Exprs], Threshold) -> boolean().
//  Determines whether a list of expressions is "smaller" than the given
//  threshold. This is largely analogous to cerl_trees:size/1 but operates on
//  our internal #iexprs{} and bails out as soon as the threshold is exceeded.
fn is_iexprs_small(exprs: &[IExpr], threshold: usize) -> bool {
    0 < is_iexprs_small_1(exprs, threshold)
}

fn is_iexprs_small_1(exprs: &[IExpr], mut threshold: usize) -> usize {
    for expr in exprs {
        if threshold == 0 {
            return 0;
        }
        threshold = is_iexprs_small_2(expr, threshold - 1);
    }

    threshold
}

fn is_iexprs_small_2(expr: &IExpr, threshold: usize) -> usize {
    match expr {
        IExpr::Try(ITry {
            ref body, handler, ..
        }) => {
            let threshold = is_iexprs_small_1(body.as_slice(), threshold);
            is_iexprs_small_2(handler.as_ref(), threshold)
        }
        IExpr::Match(IMatch { ref guards, .. }) => is_iexprs_small_1(guards.as_slice(), threshold),
        IExpr::Case(ICase { ref clauses, .. }) => {
            is_iexprs_small_iclauses(clauses.as_slice(), threshold)
        }
        IExpr::If(IIf {
            ref then_body,
            ref else_body,
            ..
        }) => {
            let threshold = is_iexprs_small_1(then_body.as_slice(), threshold);
            is_iexprs_small_1(else_body.as_slice(), threshold)
        }
        IExpr::Fun(IFun { ref clauses, .. }) => {
            is_iexprs_small_iclauses(clauses.as_slice(), threshold)
        }
        IExpr::Receive1(IReceive1 { ref clauses, .. }) => {
            is_iexprs_small_iclauses(clauses.as_slice(), threshold)
        }
        IExpr::Receive2(IReceive2 { ref clauses, .. }) => {
            is_iexprs_small_iclauses(clauses.as_slice(), threshold)
        }
        IExpr::Catch(ICatch { ref body, .. }) => is_iexprs_small_1(body.as_slice(), threshold),
        IExpr::Protect(IProtect { ref body, .. }) => is_iexprs_small_1(body.as_slice(), threshold),
        IExpr::Set(ISet { ref arg, .. }) => is_iexprs_small_2(arg.as_ref(), threshold),
        IExpr::LetRec(ILetRec { ref body, .. }) => is_iexprs_small_1(body.as_slice(), threshold),
        _ => threshold,
    }
}

fn is_iexprs_small_iclauses(clauses: &[IClause], threshold: usize) -> usize {
    let mut threshold = threshold;
    for clause in clauses {
        match is_iexprs_small_1(clause.guards.as_slice(), threshold) {
            0 => return 0,
            n => threshold = n,
        }
        match is_iexprs_small_1(clause.body.as_slice(), threshold) {
            0 => return 0,
            n => threshold = n,
        }
    }
    threshold
}

#[derive(PartialEq)]
enum BitSize {
    Undefined,
    All,
    Sized(ast::Expr),
}

fn make_bit_type(
    span: SourceSpan,
    size: Option<&ast::Expr>,
    spec: Option<BinaryEntrySpecifier>,
) -> Result<(Option<ast::Expr>, BinaryEntrySpecifier), &'static str> {
    match set_bit_type(span, size, spec)? {
        (BitSize::All, spec) => Ok((
            Some(ast::Expr::Literal(ast::Literal::Atom(Ident::new(
                symbols::All,
                span,
            )))),
            spec,
        )),
        (BitSize::Undefined, spec) => Ok((None, spec)),
        (BitSize::Sized(size), spec) => Ok((Some(size), spec)),
    }
}

fn set_bit_type(
    span: SourceSpan,
    size: Option<&ast::Expr>,
    spec: Option<BinaryEntrySpecifier>,
) -> Result<(BitSize, BinaryEntrySpecifier), &'static str> {
    let spec = spec.unwrap_or_default();
    let size = match size {
        None => BitSize::Undefined,
        Some(ast::Expr::Literal(ast::Literal::Atom(a))) if a.name == symbols::All => BitSize::All,
        Some(expr) => BitSize::Sized(expr.clone()),
    };
    match spec {
        BinaryEntrySpecifier::Binary { .. } => match size {
            BitSize::Undefined => Ok((BitSize::All, spec)),
            size => Ok((size, spec)),
        },
        BinaryEntrySpecifier::Utf8
        | BinaryEntrySpecifier::Utf16 { .. }
        | BinaryEntrySpecifier::Utf32 { .. } => match size {
            BitSize::Undefined => Ok((size, spec)),
            _ => Err("utf binary specifiers are incompatible with an explicit size"),
        },
        BinaryEntrySpecifier::Integer { .. } => match size {
            BitSize::Undefined => Ok((
                BitSize::Sized(ast::Expr::Literal(ast::Literal::Integer(span, 8.into()))),
                spec,
            )),
            BitSize::All => Err("invalid size"),
            size @ BitSize::Sized(_) => Ok((size, spec)),
        },
        BinaryEntrySpecifier::Float { .. } => match size {
            BitSize::Undefined => Ok((
                BitSize::Sized(ast::Expr::Literal(ast::Literal::Integer(span, 64.into()))),
                spec,
            )),
            BitSize::All => Err("invalid size"),
            size @ BitSize::Sized(_) => Ok((size, spec)),
        },
    }
}

fn bin_expand_string(s: Ident, mut value: Int, mut size: usize) -> Vec<ast::BinaryElement> {
    let span = s.span;
    let mut expanded = vec![];
    for c in s.as_str().get().chars().map(|c| c as i64) {
        if size >= COLLAPSE_MAX_SIZE_SEGMENT {
            expanded.push(make_combined(span, value.clone(), size));
            value = 0.into();
            size = 0;
        }
        value = value << 8;
        value = value | c;
        size += 8;
    }

    expanded.push(make_combined(span, value, size));
    expanded
}

fn make_combined(span: SourceSpan, value: Int, size: usize) -> ast::BinaryElement {
    ast::BinaryElement {
        span,
        bit_expr: ast::Expr::Literal(ast::Literal::Integer(span, value)),
        bit_size: Some(ast::Expr::Literal(ast::Literal::Integer(span, size.into()))),
        specifier: Some(BinaryEntrySpecifier::default()),
    }
}

fn verify_suitable_fields(elements: &[ast::BinaryElement]) -> Result<(), ()> {
    const MAX_UNIT: Int = Int::Small(256);

    for element in elements {
        let unit = element
            .specifier
            .as_ref()
            .map(|spec| spec.unit())
            .unwrap_or(1usize);
        match (element.bit_size.as_ref(), &element.bit_expr) {
            // utf8/16/32
            (None, ast::Expr::Literal(ast::Literal::String(_)))
            | (None, ast::Expr::Literal(ast::Literal::Char(_, _)))
            | (None, ast::Expr::Literal(ast::Literal::Integer(_, _))) => continue,
            (Some(ast::Expr::Literal(ast::Literal::Integer(_, i))), _) if i * unit >= MAX_UNIT => {
                // Always accept fields up to this size
                continue;
            }
            (
                Some(ast::Expr::Literal(ast::Literal::Integer(_, i))),
                ast::Expr::Literal(ast::Literal::Integer(_, value)),
            ) => {
                // Estimate the number of bits needed to hold the integer literal.
                // Check whether the field size is reasonable in proportion to the number
                // of bits needed.
                let size = i * unit;
                let bits_needed = value.bits();
                let bits_limit = Int::from(2 * bits_needed);
                if bits_limit >= size {
                    continue;
                }
                // More than about half of the field size will be filled out with zeros - not acceptable
                return Err(());
            }
            _ => return Err(()),
        }
    }

    Ok(())
}

fn is_new_type_test(function: Symbol, arity: u8) -> bool {
    match function {
        symbols::IsAtom
        | symbols::IsBinary
        | symbols::IsBitstring
        | symbols::IsBoolean
        | symbols::IsFloat
        | symbols::IsInteger
        | symbols::IsList
        | symbols::IsMap
        | symbols::IsNumber
        | symbols::IsPid
        | symbols::IsPort
        | symbols::IsReference
        | symbols::IsTuple => arity == 1,
        symbols::IsFunction => arity == 1 || arity == 2,
        symbols::IsRecord => arity == 2 || arity == 3,
        _ => false,
    }
}

fn is_bool_op(function: Symbol, arity: u8) -> bool {
    match function {
        symbols::Not => arity == 1,
        symbols::And | symbols::Or | symbols::Xor => arity == 2,
        _ => false,
    }
}

fn is_cmp_op(function: Symbol, arity: u8) -> bool {
    match function {
        symbols::Equal
        | symbols::NotEqual
        | symbols::Lte
        | symbols::Lt
        | symbols::Gte
        | symbols::Gt
        | symbols::EqualStrict
        | symbols::NotEqualStrict => arity == 2,
        _ => false,
    }
}

// is_guard_test(Expression) -> true | false.
//  Test if a general expression is a guard test.
//
//  Note that a local function overrides a BIF with the same name.
//  For example, if there is a local function named is_list/1,
//  any unqualified call to is_list/1 will be to the local function.
//  The guard function must be explicitly called as erlang:is_list/1.
#[inline]
fn is_guard_test(expr: &ast::Expr) -> bool {
    is_gexpr(expr)
}

fn is_type_test(fun: Symbol, arity: usize) -> bool {
    match fun {
        symbols::IsAtom
        | symbols::IsBinary
        | symbols::IsBitstring
        | symbols::IsBoolean
        | symbols::IsFloat
        | symbols::IsFunction
        | symbols::IsInteger
        | symbols::IsList
        | symbols::IsMap
        | symbols::IsNumber
        | symbols::IsPid
        | symbols::IsPort
        | symbols::IsReference
        | symbols::IsTuple
            if arity == 1 =>
        {
            true
        }
        symbols::IsFunction | symbols::IsRecord if arity == 2 => true,
        symbols::IsRecord if arity == 3 => true,
        _ => false,
    }
}

fn is_guard_bif(fun: Symbol, arity: usize) -> bool {
    match fun {
        symbols::Node | symbols::SELF if arity == 0 => true,
        symbols::Abs
        | symbols::BitSize
        | symbols::ByteSize
        | symbols::Ceil
        | symbols::Float
        | symbols::Floor
        | symbols::Hd
        | symbols::Length
        | symbols::MapSize
        | symbols::Node
        | symbols::Round
        | symbols::Size
        | symbols::Tl
        | symbols::Trunc
        | symbols::TupleSize
            if arity == 1 =>
        {
            true
        }
        symbols::BinaryPart | symbols::Element | symbols::IsMapKey | symbols::MapGet
            if arity == 2 =>
        {
            true
        }
        symbols::BinaryPart if arity == 3 => true,
        _ => is_type_test(fun, arity),
    }
}

fn is_gexpr(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::Var(_) => true,
        ast::Expr::Literal(_) => true,
        ast::Expr::Cons(cons) => is_gexpr(&cons.head) && is_gexpr(&cons.tail),
        ast::Expr::Tuple(tup) => is_gexpr_list(tup.elements.as_slice()),
        ast::Expr::Map(map) => is_map_fields(map.fields.as_slice()),
        ast::Expr::MapUpdate(mu) => {
            is_gexpr(mu.map.as_ref()) && is_map_fields(mu.updates.as_slice())
        }
        ast::Expr::Binary(bin) => bin.elements.iter().all(|e| {
            is_gexpr(&e.bit_expr) && e.bit_size.as_ref().map(|sz| is_gexpr(sz)).unwrap_or(true)
        }),
        ast::Expr::Apply(ast::Apply { callee, args, .. }) => match callee.as_ref() {
            ast::Expr::FunctionVar(name) => match (name.module(), name.function()) {
                (Some(symbols::Erlang), Some(f)) => {
                    let arity = args.len();
                    is_gexpr_op(f, arity) && is_gexpr_list(args.as_slice())
                }
                _ => false,
            },
            _ => false,
        },
        ast::Expr::BinaryExpr(ast::BinaryExpr { op, lhs, rhs, .. }) if op.is_guard_op() => {
            is_gexpr(lhs.as_ref()) && is_gexpr(rhs.as_ref())
        }
        ast::Expr::UnaryExpr(ast::UnaryExpr { op, operand, .. }) if op.is_guard_op() => {
            is_gexpr(operand.as_ref())
        }
        ast::Expr::FunctionVar(_)
        | ast::Expr::DelayedSubstitution(_)
        | ast::Expr::Record(_)
        | ast::Expr::RecordAccess(_)
        | ast::Expr::RecordIndex(_)
        | ast::Expr::RecordUpdate(_)
        | ast::Expr::Generator(_) => unreachable!(),
        _ => false,
    }
}

fn is_gexpr_op(op: Symbol, arity: usize) -> bool {
    match arity {
        1 => match UnaryOp::from_symbol(op).map(|o| o.is_guard_op()) {
            Ok(result) => result,
            Err(_) => is_guard_bif(op, arity),
        },
        2 => match BinaryOp::from_symbol(op).map(|o| o.is_guard_op()) {
            Ok(result) => result,
            Err(_) => is_guard_bif(op, arity),
        },
        _ => is_guard_bif(op, arity),
    }
}

#[inline]
fn is_gexpr_list(elements: &[ast::Expr]) -> bool {
    elements.iter().all(is_gexpr)
}

fn is_map_fields(fields: &[ast::MapField]) -> bool {
    for field in fields {
        if !is_gexpr(field.key_ref()) {
            return false;
        }
        if !is_gexpr(field.value_ref()) {
            return false;
        }
    }
    true
}

#[derive(PartialEq, Eq)]
enum MapSortKey {
    Lit(Lit),
    Var(Symbol),
    Size(usize),
}
impl PartialOrd for MapSortKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MapSortKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        match (self, other) {
            (Self::Lit(x), Self::Lit(y)) => x.cmp(y),
            (Self::Lit(_), _) => Ordering::Less,
            (Self::Var(x), Self::Var(y)) => x.cmp(y),
            (Self::Var(_), Self::Lit(_)) => Ordering::Greater,
            (Self::Var(_), Self::Size(_)) => Ordering::Less,
            (Self::Size(x), Self::Size(y)) => x.cmp(y),
            (Self::Size(_), _) => Ordering::Greater,
        }
    }
}

fn map_sort_key(key: &[IExpr], keymap: &BTreeMap<MapSortKey, Vec<IMapPair>>) -> MapSortKey {
    if key.len() != 1 {
        return MapSortKey::Size(keymap.len());
    }
    match &key[0] {
        IExpr::Literal(Literal { value, .. }) => MapSortKey::Lit(value.clone()),
        IExpr::Var(var) => MapSortKey::Var(var.name()),
        _other => MapSortKey::Size(keymap.len()),
    }
}

fn force_safe(context: &mut FunctionContext, expr: IExpr) -> (IExpr, Vec<IExpr>) {
    match expr {
        IExpr::Match(imatch) => {
            let (le, mut pre) = force_safe(context, *imatch.arg);

            // Make sure we don't duplicate the expression E
            match le {
                le @ IExpr::Var(_) => {
                    // Le is a variable
                    // Thus: P = Le, Le.
                    pre.push(IExpr::Match(IMatch {
                        span: imatch.span,
                        annotations: imatch.annotations,
                        pattern: imatch.pattern,
                        guards: imatch.guards,
                        arg: Box::new(le.clone()),
                        fail: imatch.fail,
                    }));
                    (le, pre)
                }
                le => {
                    // Le is not a variable.
                    // Thus: NewVar = P = Le, NewVar.
                    let v = context.next_var(Some(le.span()));
                    let pattern = imatch.pattern;
                    let pattern = IExpr::Alias(IAlias {
                        span: pattern.span(),
                        annotations: Annotations::default(),
                        var: v.clone(),
                        pattern,
                    });
                    pre.push(IExpr::Match(IMatch {
                        span: imatch.span,
                        annotations: imatch.annotations,
                        pattern: Box::new(pattern),
                        guards: imatch.guards,
                        arg: Box::new(le),
                        fail: imatch.fail,
                    }));
                    (IExpr::Var(v), pre)
                }
            }
        }
        expr if expr.is_safe() => (expr, vec![]),
        expr => {
            let span = expr.span();
            let v = context.next_var(Some(span));
            let var = IExpr::Var(v.clone());
            (var, vec![IExpr::Set(ISet::new(span, v, expr))])
        }
    }
}
