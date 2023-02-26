use std::cell::UnsafeCell;
use std::rc::Rc;

use rpds::{rbt_set, RedBlackTreeSet};

use firefly_diagnostics::*;
use firefly_intern::{symbols, Ident, Symbol};
use firefly_pass::Pass;
use firefly_syntax_base::*;

use crate::{passes::FunctionContext, *};

/// Phase 3: Rewrite clauses to make implicit exports explicit
///
/// Step "backwards" over icore code using variable usage
/// annotations to change implicit exported variables to explicit
/// returns.
pub struct RewriteExports {
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl RewriteExports {
    pub fn new(context: Rc<UnsafeCell<FunctionContext>>) -> Self {
        Self { context }
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
impl Pass for RewriteExports {
    type Input<'a> = IFun;
    type Output<'a> = Fun;

    fn run<'a>(&mut self, ifun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        match self.cexpr(IExpr::Fun(ifun), rbt_set![]) {
            Ok((box Expr::Fun(fun), _, _)) if self.context().is_nif => {
                /* TODO: the Erlang compiler injects a nif_start primop, need to explore how that's used
                let span = fun.span;
                let body = fun.body;
                let body_span = body.span();
                let body = Box::new(Expr::Seq(Seq {
                    span: body_span,
                    annotations: Annotations::default(),
                    arg: Box::new(Expr::PrimOp(PrimOp::new(
                        body_span,
                        symbols::NifStart,
                        vec![],
                    ))),
                    body,
                }));
                Ok(Fun {
                    span,
                    annotations: fun.annotations,
                    name: fun.name,
                    vars: fun.vars,
                    body,
                })
                */
                Ok(fun)
            }
            Ok((box Expr::Fun(fun), _, _)) => Ok(fun),
            Ok(ref expr) => panic!("unexpected expr result, expected fun: {:#?}", expr),
            Err(reason) => Err(reason),
        }
    }
}

impl RewriteExports {
    // cclause(Lclause, [AfterVar], State) -> {Cclause,State}.
    //  The AfterVars are the exported variables.
    fn cclause(
        &mut self,
        clause: IClause,
        exports: RedBlackTreeSet<Ident>,
    ) -> anyhow::Result<Clause> {
        let span = clause.span;
        let patterns = self.cpattern_list(clause.patterns)?;
        let (body, _) = self.cexprs(clause.body, exports)?;
        let guard = self.cguard(clause.guards)?;
        Ok(Clause {
            span,
            annotations: clause.annotations,
            patterns,
            guard,
            body,
        })
    }

    fn cclauses(
        &mut self,
        mut clauses: Vec<IClause>,
        exports: RedBlackTreeSet<Ident>,
    ) -> anyhow::Result<Vec<Clause>> {
        clauses
            .drain(..)
            .map(|clause| self.cclause(clause, exports.clone()))
            .try_collect()
    }

    fn cguard(&mut self, guards: Vec<IExpr>) -> anyhow::Result<Option<Box<Expr>>> {
        if guards.is_empty() {
            return Ok(None);
        }
        let (guard, _) = self.cexprs(guards, rbt_set![])?;
        Ok(Some(guard))
    }

    fn cpattern_list(&mut self, mut patterns: Vec<IExpr>) -> anyhow::Result<Vec<Expr>> {
        patterns
            .drain(..)
            .map(|pat| self.cpattern(pat))
            .try_collect()
    }

    fn cpattern(&mut self, pattern: IExpr) -> anyhow::Result<Expr> {
        match pattern {
            IExpr::Alias(IAlias {
                span,
                annotations,
                var,
                pattern,
            }) => {
                let pattern = self.cpattern(*pattern)?;
                Ok(Expr::Alias(Alias {
                    span,
                    annotations,
                    var,
                    pattern: Box::new(pattern),
                }))
            }
            IExpr::Cons(ICons {
                span,
                annotations,
                head,
                tail,
            }) => {
                let head = self.cpattern(*head)?;
                let tail = self.cpattern(*tail)?;
                Ok(Expr::Cons(Cons {
                    span,
                    annotations,
                    head: Box::new(head),
                    tail: Box::new(tail),
                }))
            }
            IExpr::Tuple(ITuple {
                span,
                annotations,
                elements,
            }) => {
                let elements = self.cpattern_list(elements)?;
                Ok(Expr::Tuple(Tuple {
                    span,
                    annotations,
                    elements,
                }))
            }
            IExpr::Map(IMap {
                span,
                annotations,
                arg,
                pairs,
                ..
            }) => {
                let arg = self.cpattern(*arg)?;
                let pairs = self.cpattern_map_pairs(pairs)?;
                Ok(Expr::Map(Map {
                    span,
                    annotations,
                    arg: Box::new(arg),
                    pairs,
                    is_pattern: true,
                }))
            }
            IExpr::Binary(IBinary {
                span,
                annotations,
                mut segments,
            }) => {
                let segments = segments
                    .drain(..)
                    .map(|seg| self.cpattern_bin_segment(seg))
                    .try_collect()?;
                Ok(Expr::Binary(Binary {
                    span,
                    annotations,
                    segments,
                }))
            }
            IExpr::Literal(lit) => Ok(Expr::Literal(lit)),
            IExpr::Var(v) => Ok(Expr::Var(v)),
            other => unimplemented!("unhandled icore pattern expression: {:#?}", &other),
        }
    }

    fn cpattern_map_pairs(&mut self, mut pairs: Vec<IMapPair>) -> anyhow::Result<Vec<MapPair>> {
        pairs
            .drain(..)
            .map(|IMapPair { op, key, value }| {
                let (key, _) = self.cexprs(key, rbt_set![])?;
                let value = self.cpattern(*value)?;
                Ok(MapPair {
                    op,
                    key,
                    value: Box::new(value),
                })
            })
            .try_collect()
    }

    fn cpattern_bin_segment(&mut self, segment: IBitstring) -> anyhow::Result<Bitstring> {
        let (value, _, _) = self.cexpr(*segment.value, rbt_set![])?;
        let size = if segment.size.is_empty() {
            None
        } else {
            let (sz, _) = self.cexprs(segment.size, rbt_set![])?;
            Some(sz)
        };
        Ok(Bitstring {
            span: segment.span,
            annotations: segment.annotations,
            size,
            value,
            spec: segment.spec,
        })
    }

    // cexprs([Lexpr], [AfterVar], State) -> {Cexpr,[AfterVar],State}.
    //  Must be sneaky here at the last expr when combining exports for the
    //  whole sequence and exports for that expr.
    fn cexprs(
        &mut self,
        mut iexprs: Vec<IExpr>,
        exports: RedBlackTreeSet<Ident>,
    ) -> anyhow::Result<(Box<Expr>, RedBlackTreeSet<Ident>)> {
        assert!(!iexprs.is_empty());
        let mut iter = iexprs.drain(..).peekable();
        let expr = iter.next().unwrap();
        let is_last = iter.peek().is_none();
        match expr {
            IExpr::Set(set) if is_last => {
                // Make return value explicit, and make Var true top level.
                let mut annotations = Annotations::new();
                annotations.insert_mut(symbols::Used, rbt_set![set.var.name]);
                let simple = IExpr::Simple(ISimple {
                    annotations,
                    expr: Box::new(IExpr::Var(set.var.clone())),
                });
                self.cexprs(vec![IExpr::Set(set), simple], exports)
            }
            IExpr::Set(ISet {
                span,
                annotations,
                var,
                arg,
            }) => {
                let (body, exports) = self.cexprs(iter.collect(), exports)?;
                let (arg, es, us) = self.cexpr(*arg, exports.clone())?;
                let exports = sets::union(us, exports);
                let vars = core::iter::once(var.name)
                    .chain(es.iter().copied())
                    .map(|v| Var::new(v))
                    .collect();
                Ok((
                    Box::new(Expr::Let(Let {
                        span,
                        annotations,
                        vars,
                        arg,
                        body,
                    })),
                    exports,
                ))
            }
            iexpr if is_last => {
                let (expr, es, us) = self.cexpr(iexpr, exports.clone())?;
                let mut exp = exports
                    .iter()
                    .copied()
                    .map(|id| Var::new(id))
                    .collect::<Vec<_>>();
                let span = expr.span();
                // The export variables
                let exports = sets::union(us, exports);
                if es.is_empty() {
                    let values = core::iter::once(*expr)
                        .chain(exp.drain(..).map(Expr::Var))
                        .collect();
                    Ok((Box::new(Values::new(span, values)), exports))
                } else {
                    let rname = self.context_mut().next_var(Some(span));
                    let r = Expr::Var(rname.clone());
                    let values = core::iter::once(r.clone())
                        .chain(exp.drain(..).map(Expr::Var))
                        .collect();
                    let vars = core::iter::once(rname)
                        .chain(es.iter().copied().map(|v| Var::new(v)))
                        .collect();
                    let body = Box::new(Values::new(span, values));
                    Ok((
                        Box::new(Expr::Let(Let {
                            span,
                            annotations: Annotations::default(),
                            vars,
                            arg: expr,
                            body,
                        })),
                        exports,
                    ))
                }
            }
            other => {
                let (body, exports) = self.cexprs(iter.collect(), exports)?;
                let (arg, es, us) = self.cexpr(other, exports.clone())?;
                let exports = sets::union(us, exports);
                let span = arg.span();
                let annotations = Annotations::default();
                if es.is_empty() {
                    Ok((
                        Box::new(Expr::Seq(Seq {
                            span,
                            annotations,
                            arg,
                            body,
                        })),
                        exports,
                    ))
                } else {
                    let r = self.context_mut().next_var(Some(span));
                    let vars = core::iter::once(r.name)
                        .chain(es.iter().copied())
                        .map(|v| Var::new(v))
                        .collect();
                    Ok((
                        Box::new(Expr::Let(Let {
                            span,
                            annotations,
                            vars,
                            arg,
                            body,
                        })),
                        exports,
                    ))
                }
            }
        }
    }

    // cexpr(Lexpr, [AfterVar], State) -> {Cexpr,[ExpVar],[UsedVar],State}.
    fn cexpr(
        &mut self,
        iexpr: IExpr,
        exports: RedBlackTreeSet<Ident>,
    ) -> anyhow::Result<(Box<Expr>, RedBlackTreeSet<Ident>, RedBlackTreeSet<Ident>)> {
        match iexpr {
            IExpr::LetRec(mut lexpr) => {
                let used = lexpr.used_vars();
                let exported = sets::intersection(lexpr.new_vars(), exports.clone());
                let mut defs = Vec::with_capacity(lexpr.defs.len());
                let mut us = rbt_set![];
                for (var, def) in lexpr.defs.drain(..) {
                    let (f, _, us1) = self.cexpr(def, rbt_set![])?;
                    defs.push((var, *f));
                    us = sets::union(us1, us.clone());
                }
                let (body, _) = self.cexprs(lexpr.body, exported.clone())?;
                let lexpr = Box::new(Expr::LetRec(LetRec {
                    span: lexpr.span,
                    annotations: lexpr.annotations,
                    defs,
                    body,
                }));
                Ok((lexpr, exported, used))
            }
            IExpr::Case(mut case) => {
                let span = case.span;
                let used = case.used_vars();
                let exported = sets::intersection(case.new_vars(), exports.clone());
                let args = case
                    .args
                    .drain(..)
                    .map(|arg| self.cexpr(arg, exports.clone()).map(|(e, _, _)| *e))
                    .try_collect()?;
                let mut clauses = self.cclauses(case.clauses, exported.clone())?;
                let fail = self.cclause(*case.fail, rbt_set![])?;
                let fail = self.add_dummy_export(fail, exported.clone());
                clauses.push(fail);
                let cexpr = Box::new(Expr::Case(Case {
                    span,
                    annotations: case.annotations,
                    arg: Box::new(Values::new(span, args)),
                    clauses,
                }));
                Ok((cexpr, exported, used))
            }
            IExpr::If(expr) => {
                let span = expr.span;
                let used = expr.used_vars();
                let exported = sets::intersection(expr.new_vars(), exports.clone());
                let (guard, _) = self.cexprs(expr.guards, rbt_set![])?;
                let (then_body, _) = self.cexprs(expr.then_body, exported.clone())?;
                let (else_body, _) = self.cexprs(expr.else_body, exported.clone())?;

                let cexpr = Box::new(Expr::If(If {
                    span,
                    annotations: expr.annotations,
                    guard,
                    then_body,
                    else_body,
                }));
                Ok((cexpr, exported, used))
            }
            IExpr::Receive1(recv) => {
                let span = recv.span;
                let used = recv.used_vars();
                let exported = sets::intersection(recv.new_vars(), exports);
                let clauses = self.cclauses(recv.clauses, exported.clone())?;
                let t = Expr::Literal(Literal::atom(recv.span, symbols::True));
                let action = (0..=exported.size()).map(|_| t.clone()).collect();
                let rexpr = Box::new(Expr::Receive(Receive {
                    span,
                    annotations: recv.annotations,
                    clauses,
                    timeout: Box::new(Expr::Literal(Literal::atom(span, symbols::Infinity))),
                    action: Box::new(Values::new(span, action)),
                }));
                Ok((rexpr, exported, used))
            }
            IExpr::Receive2(recv) => {
                let used = recv.used_vars();
                let exported = sets::intersection(recv.new_vars(), exports.clone());
                let (timeout, _, _) = self.cexpr(*recv.timeout, exports.clone())?;
                let clauses = self.cclauses(recv.clauses, exported.clone())?;
                let (action, _) = self.cexprs(recv.action, exported.clone())?;
                let rexpr = Box::new(Expr::Receive(Receive {
                    span: recv.span,
                    annotations: recv.annotations,
                    clauses,
                    timeout,
                    action,
                }));
                Ok((rexpr, exported, used))
            }
            IExpr::Try(mut texpr) => {
                // No variables are exported from try/catch. Starting in OTP 24,
                // variables bound in the argument (the code between the 'try' and
                // the 'of' keywords) are exported to the body (the code following
                // the 'of' keyword).
                let used = texpr.used_vars();
                let exported = sets::intersection(
                    sets::new_in_any(texpr.args.iter()),
                    sets::used_in_any(texpr.body.iter()),
                );
                let (arg, _) = self.cexprs(texpr.args, exported.clone())?;
                let (body, _) = self.cexprs(texpr.body, rbt_set![])?;
                let (handler, _, _) = self.cexpr(*texpr.handler, rbt_set![])?;
                let vars = texpr
                    .vars
                    .drain(..)
                    .chain(exported.iter().copied().map(|i| Var::new(i)))
                    .collect();
                let texpr = Box::new(Expr::Try(Try {
                    span: texpr.span,
                    annotations: texpr.annotations,
                    arg,
                    vars,
                    body,
                    evars: texpr.evars,
                    handler,
                }));
                Ok((texpr, rbt_set![], used))
            }
            IExpr::Catch(ICatch {
                span,
                annotations,
                body,
            }) => {
                // Never exports
                let used = annotations.used_vars().unwrap_or_default();
                let (body, _) = self.cexprs(body, rbt_set![])?;
                Ok((
                    Box::new(Expr::Catch(Catch {
                        span,
                        annotations,
                        body,
                    })),
                    rbt_set![],
                    used,
                ))
            }
            IExpr::Fun(fun) if fun.name.is_none() => self
                .cfun(fun)
                .map(|(f, exp, us)| (Box::new(Expr::Fun(f)), exp, us)),
            IExpr::Fun(mut fun) => {
                let span = fun.span;
                let us0 = fun.used_vars();
                let name = fun.name.clone().unwrap();
                if !us0.contains(&name) {
                    self.cfun(fun)
                        .map(|(f, exp, us)| (Box::new(Expr::Fun(f)), exp, us))
                } else {
                    let recvar = Var::new_with_arity(name, fun.fail.patterns.len());
                    fun.annotations.remove_mut(name.name);
                    let (fun, _, us1) = self.cfun(fun)?;
                    let lexpr = Box::new(Expr::Let(Let {
                        span,
                        annotations: Annotations::default(),
                        vars: vec![Var::new(name)],
                        arg: Box::new(Expr::Var(recvar.clone())),
                        body: fun.body,
                    }));
                    let annotations = fun.annotations;
                    let fun = Expr::Fun(Fun {
                        span,
                        annotations: annotations.clone(),
                        name: fun.name,
                        vars: fun.vars,
                        body: lexpr,
                    });
                    let body = Box::new(Expr::Var(recvar.clone()));
                    let lrexpr = Box::new(Expr::LetRec(LetRec {
                        span,
                        annotations,
                        defs: vec![(recvar, fun)],
                        body,
                    }));
                    Ok((lrexpr, rbt_set![], us1))
                }
            }
            IExpr::Apply(IApply {
                span,
                annotations,
                callee,
                mut args,
            }) => {
                let (callee, _) = self.cexprs(callee, rbt_set![])?;
                let args = args
                    .drain(..)
                    .map(|a| self.cexpr(a, rbt_set![]).map(|(e, _, _)| *e))
                    .try_collect()?;
                let used = annotations.used_vars().unwrap_or_default();
                let apply = Apply {
                    span,
                    annotations,
                    callee,
                    args,
                };
                Ok((Box::new(Expr::Apply(apply)), rbt_set![], used))
            }
            IExpr::Call(ICall {
                span,
                annotations,
                module,
                function,
                mut args,
            }) => {
                let (module, _, _) = self.cexpr(*module, rbt_set![])?;
                let (function, _, _) = self.cexpr(*function, rbt_set![])?;
                let args = args
                    .drain(..)
                    .map(|a| self.cexpr(a, rbt_set![]).map(|(e, _, _)| *e))
                    .try_collect()?;
                let used = annotations.used_vars().unwrap_or_default();
                let call = Call {
                    span,
                    annotations,
                    module,
                    function,
                    args,
                };
                Ok((Box::new(Expr::Call(call)), rbt_set![], used))
            }
            IExpr::PrimOp(IPrimOp {
                span,
                annotations,
                name,
                mut args,
            }) => {
                let args = args
                    .drain(..)
                    .map(|a| self.cexpr(a, rbt_set![]).map(|(e, _, _)| *e))
                    .try_collect()?;
                let used = annotations.used_vars().unwrap_or_default();
                let op = PrimOp {
                    span,
                    annotations,
                    name,
                    args,
                };
                Ok((Box::new(Expr::PrimOp(op)), rbt_set![], used))
            }
            IExpr::Protect(IProtect {
                span,
                annotations,
                body,
            }) => {
                let used = annotations.used_vars().unwrap_or_default();
                let (body, _) = self.cexprs(body, rbt_set![])?;
                // Name doesn't matter here
                let v = Var::new(Ident::new(Symbol::intern("Try"), span));
                let t = Var::new(Ident::new(symbols::Underscore, span));
                let r = Var::new(Ident::new(symbols::Underscore, span));
                let texpr = Try {
                    span,
                    annotations,
                    arg: body,
                    vars: vec![v.clone()],
                    body: Box::new(Expr::Var(v)),
                    evars: vec![t, r],
                    handler: Box::new(Expr::Literal(Literal::atom(span, symbols::False))),
                };
                Ok((Box::new(Expr::Try(texpr)), rbt_set![], used))
            }
            IExpr::Binary(IBinary {
                span,
                annotations,
                mut segments,
            }) => {
                let used = annotations.used_vars().unwrap_or_default();
                let segments = segments
                    .drain(..)
                    .map(|segment| {
                        let (value, _, _) = self.cexpr(*segment.value, rbt_set![])?;
                        let size = if segment.size.is_empty() {
                            None
                        } else {
                            let (size, _) = self.cexprs(segment.size, rbt_set![])?;
                            Some(size)
                        };
                        Ok::<_, anyhow::Error>(Bitstring {
                            span: segment.span,
                            annotations: segment.annotations,
                            value,
                            size,
                            spec: segment.spec,
                        })
                    })
                    .try_collect()?;
                let bin = Binary {
                    span,
                    annotations,
                    segments,
                };
                Ok((Box::new(Expr::Binary(bin)), rbt_set![], used))
            }
            IExpr::Literal(lit) => {
                let used = lit.used_vars();
                Ok((Box::new(Expr::Literal(lit)), rbt_set![], used))
            }
            IExpr::Simple(ISimple { expr, .. }) => {
                let used = expr.used_vars();
                assert!(expr.is_simple());
                let (expr, _, _) = self.cexpr(*expr, rbt_set![])?;
                Ok((expr, rbt_set![], used))
            }
            IExpr::Cons(ICons {
                span,
                annotations,
                head,
                tail,
            }) => {
                let used = annotations.used_vars().unwrap_or_default();
                let (head, _, _) = self.cexpr(*head, rbt_set![])?;
                let (tail, _, _) = self.cexpr(*tail, rbt_set![])?;
                let cons = Box::new(Expr::Cons(Cons {
                    span,
                    annotations,
                    head,
                    tail,
                }));
                Ok((cons, rbt_set![], used))
            }
            IExpr::Tuple(ITuple {
                span,
                annotations,
                mut elements,
            }) => {
                let elements = elements
                    .drain(..)
                    .map(|e| self.cexpr(e, rbt_set![]).map(|(e, _, _)| *e))
                    .try_collect()?;
                let used = annotations.used_vars().unwrap_or_default();
                let tuple = Box::new(Expr::Tuple(Tuple {
                    span,
                    annotations,
                    elements,
                }));
                Ok((tuple, rbt_set![], used))
            }
            IExpr::Map(mut map) if map.is_simple() => {
                assert_eq!(map.is_pattern, false);
                let used = map.used_vars();
                let (arg, _, _) = self.cexpr(*map.arg, rbt_set![])?;
                let pairs = map
                    .pairs
                    .drain(..)
                    .map(|mut pair| {
                        let (key, _, _) = self.cexpr(pair.key.pop().unwrap(), rbt_set![])?;
                        let (value, _, _) = self.cexpr(*pair.value, rbt_set![])?;
                        Ok::<_, anyhow::Error>(MapPair {
                            op: pair.op,
                            key,
                            value,
                        })
                    })
                    .try_collect::<Vec<_>>()?;
                let map = Box::new(Expr::Map(Map {
                    span: map.span,
                    annotations: map.annotations,
                    arg,
                    pairs,
                    is_pattern: false,
                }));
                Ok((map, rbt_set![], used))
            }
            IExpr::Var(var) => {
                let used = var.used_vars();
                Ok((Box::new(Expr::Var(var)), rbt_set![], used))
            }
            iexpr => panic!("unexpected icore expression: {:#?}", &iexpr),
        }
    }

    fn cfun(
        &mut self,
        ifun: IFun,
    ) -> anyhow::Result<(Fun, RedBlackTreeSet<Ident>, RedBlackTreeSet<Ident>)> {
        let span = ifun.span;
        let mut annotations = ifun.annotations;
        let used = annotations.used_vars().unwrap_or_default();

        let id = ifun.id.unwrap();
        annotations.insert_mut(symbols::Id, Literal::atom(SourceSpan::default(), id.name));

        let vars = ifun.vars;
        let values = vars.iter().map(|v| Expr::Var(v.clone())).collect();

        let mut clauses = self.cclauses(ifun.clauses, rbt_set![])?;
        let fail = self.cclause(*ifun.fail, rbt_set![])?;
        clauses.push(fail);

        let body = Box::new(Expr::Case(Case {
            span,
            annotations: annotations.clone(),
            arg: Box::new(Values::new(span, values)),
            clauses,
        }));
        let fun = Fun {
            span,
            name: id.name,
            annotations,
            vars,
            body,
        };
        Ok((fun, rbt_set![], used))
    }

    fn add_dummy_export(&mut self, clause: Clause, exports: RedBlackTreeSet<Ident>) -> Clause {
        if exports.is_empty() {
            return clause;
        }

        // Add dummy export in order to always return the correct number
        // of values for the default clause.
        let span = clause.span;
        let v = self.context_mut().next_var(Some(span));
        let values = core::iter::once(Expr::Var(v.clone()))
            .chain(exports.iter().map(|_| Expr::Literal(Literal::nil(span))))
            .collect();
        let body = Box::new(Values::new(span, values));
        let lexpr = Box::new(Expr::Let(Let {
            span,
            annotations: Annotations::default(),
            arg: clause.body,
            vars: vec![v],
            body,
        }));
        Clause {
            span,
            annotations: clause.annotations,
            patterns: clause.patterns,
            guard: clause.guard,
            body: lexpr,
        }
    }
}
