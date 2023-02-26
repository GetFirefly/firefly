use std::cell::UnsafeCell;
use std::rc::Rc;

use rpds::{rbt_set, RedBlackTreeSet};

use firefly_diagnostics::*;
use firefly_intern::{symbols, Ident};
use firefly_pass::Pass;
use firefly_syntax_base::*;

use super::Known;
use crate::{passes::FunctionContext, *};

/// Phase 2: Annotate variable usage
///
/// Step "forwards" over the icore code annotating each "top-level"
/// thing with variable usage.  Detect bound variables in matching
/// and replace with explicit guard test.  Annotate "internal-core"
/// expressions with variables they use and create.  Convert matches
/// to cases when not pure assignments.
pub struct AnnotateVariableUsage {
    context: Rc<UnsafeCell<FunctionContext>>,
}
impl AnnotateVariableUsage {
    pub fn new(context: Rc<UnsafeCell<FunctionContext>>) -> Self {
        Self { context }
    }

    #[inline(always)]
    fn context_mut(&self) -> &mut FunctionContext {
        unsafe { &mut *self.context.get() }
    }
}
impl Pass for AnnotateVariableUsage {
    type Input<'a> = IFun;
    type Output<'a> = IFun;

    fn run<'a>(&mut self, ifun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        match self.uexpr(IExpr::Fun(ifun), Known::default()) {
            Ok(IExpr::Fun(ifun)) => Ok(ifun),
            Ok(_) => panic!("unexpected iexpr result, expected ifun"),
            Err(reason) => Err(reason),
        }
    }
}

impl AnnotateVariableUsage {
    fn ufun_clauses(
        &mut self,
        mut clauses: Vec<IClause>,
        known: Known,
    ) -> anyhow::Result<Vec<IClause>> {
        clauses
            .drain(..)
            .map(|c| self.ufun_clause(c, known.clone()))
            .try_collect()
    }

    fn ufun_clause(&mut self, clause: IClause, known: Known) -> anyhow::Result<IClause> {
        // Since variables in fun heads shadow previous variables
        // with the same name, we used to send an empty list as the
        // known variables when doing liveness analysis of the patterns
        // (in the upattern functions).
        //
        // With the introduction of expressions in size for binary
        // segments and in map keys, all known variables must be
        // available when analysing those expressions, or some variables
        // might not be seen as used if, for example, the expression includes
        // a case construct.
        //
        // Therefore, we will send in the complete list of known variables
        // when doing liveness analysis of patterns. This is
        // safe because any shadowing variables in a fun head has
        // been renamed.
        let (mut clause, pvs, used, _new) = self.do_uclause(clause, known)?;
        let used = sets::subtract(used, pvs);
        clause.annotate(symbols::Used, used);
        clause.annotate(symbols::New, rbt_set![]);
        Ok(clause)
    }

    fn uclauses(
        &mut self,
        mut clauses: Vec<IClause>,
        known: Known,
    ) -> anyhow::Result<Vec<IClause>> {
        clauses
            .drain(..)
            .map(|c| self.uclause(c, known.clone()))
            .try_collect()
    }

    fn uclause(&mut self, clause: IClause, known: Known) -> anyhow::Result<IClause> {
        let (mut clause, _pvs, used, new) = self.do_uclause(clause, known)?;
        clause.annotate(symbols::Used, used);
        clause.annotate(symbols::New, new);
        Ok(clause)
    }

    fn do_uclause(
        &mut self,
        clause: IClause,
        known: Known,
    ) -> anyhow::Result<(
        IClause,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
    )> {
        let mut annotations = clause.annotations;
        let (patterns, guards, pvs, used) = self.upattern_list(clause.patterns, known.clone())?;
        let is_skip_clause = annotations.contains(symbols::SkipClause);
        let pguards = if is_skip_clause {
            // This is the skip clause for a binary generator.
            // To ensure that it will properly skip the nonmatching
            // patterns in generators such as:
            //
            //     <<V, V>> <= Gen
            //
            // we must remove any generated pre guard.
            annotations.remove_mut(symbols::SkipClause);
            vec![]
        } else {
            guards
        };
        let used = sets::union(
            used.clone(),
            sets::intersection(pvs.clone(), known.get().clone()),
        );
        let new = sets::subtract(pvs.clone(), used.clone());
        let known1 = known.union(&new);
        let guards = self.uguard(pguards, clause.guards, known1.clone())?;
        let used_in_guard = sets::used_in_any(guards.iter());
        let new_in_guard = sets::new_in_any(guards.iter());
        let known2 = known1.union(&new_in_guard);

        // Consider this example:
        //
        //     {X = A,
        //      begin X = B, fun() -> X = C end() end}.
        //
        // At this point it has been rewritten to something similar
        // like this (the fun body has not been rewritten yet):
        //
        //     {X = A,
        //      begin
        //           X1 = B,
        //           if
        //             X1 =:= X -> ok;
        //              true -> error({badmatch,X1})
        //           end,
        //           fun() -> ... end() end
        //      end}.
        //
        // In this example, the variable `X` is a known variable that must
        // be passed into the fun body (because of `X = B` above). To ensure
        // that it is, we must call known_bind/2 with the variables used
        // in the guard (`X1` and `X`; any variables used must surely be
        // bound).

        let known3 = known2.bind(&used_in_guard);
        let (body, _) = self.uexprs(clause.body, known3)?;
        let used = sets::intersection(
            sets::union(
                used,
                sets::union(used_in_guard, sets::used_in_any(body.iter())),
            ),
            known.get().clone(),
        );
        let new = sets::union(
            new,
            sets::union(new_in_guard, sets::new_in_any(body.iter())),
        );
        let clause = IClause {
            span: clause.span,
            annotations,
            patterns,
            guards,
            body,
        };
        Ok((clause, pvs, used, new))
    }

    // uguard([Test], [Kexpr], [KnownVar], State) -> {[Kexpr],State}.
    //  Build a guard expression list by folding in the equality tests.
    fn uguard(
        &mut self,
        mut tests: Vec<IExpr>,
        guards: Vec<IExpr>,
        known: Known,
    ) -> anyhow::Result<Vec<IExpr>> {
        if tests.is_empty() && guards.is_empty() {
            return Ok(vec![]);
        }
        if guards.is_empty() {
            // No guard, so fold together equality tests
            let last = tests.pop().unwrap();
            return self.uguard(tests, vec![last], known);
        }
        // `guards` must contain at least one element here
        let guards = tests.drain(..).rfold(guards, |mut gs, test| {
            let span = test.span();
            let l = self.context_mut().next_var(Some(span));
            let r = self.context_mut().next_var(Some(span));
            let last = gs.pop().unwrap();
            let setl = IExpr::Set(ISet::new(span, l.clone(), test));
            let setr = IExpr::Set(ISet::new(span, r.clone(), last));
            let call = IExpr::Call(ICall::new(
                span,
                symbols::Erlang,
                symbols::And,
                vec![IExpr::Var(l), IExpr::Var(r)],
            ));
            gs.insert(0, setl);
            gs.push(setr);
            gs.push(call);
            gs
        });
        self.uexprs(guards, known).map(|(gs, _)| gs)
    }

    fn uexprs(
        &mut self,
        mut exprs: Vec<IExpr>,
        mut known: Known,
    ) -> anyhow::Result<(Vec<IExpr>, Known)> {
        let mut result = Vec::with_capacity(exprs.len());
        let mut new = rbt_set![];
        let mut iter = exprs.drain(..).peekable();
        while let Some(expr) = iter.next() {
            let is_last = iter.peek().is_none();
            match expr {
                IExpr::Exprs(IExprs { mut bodies, .. }) => {
                    // Need to effectively linearize the expressions in the bodies while
                    // properly tracking known bindings for the body/group
                    known.start_group();
                    for body in bodies.drain(..) {
                        let (mut body, known1) = self.uexprs(body, known.clone())?;
                        known = known1;
                        result.append(&mut body);
                        known.end_body();
                    }
                    known.end_group();
                }
                IExpr::Match(IMatch {
                    span,
                    annotations,
                    pattern,
                    arg,
                    fail,
                    ..
                }) => {
                    match *pattern {
                        IExpr::Var(var) if !known.contains(&var.name) => {
                            // Assignment to new variable
                            let expr =
                                self.uexpr(IExpr::Set(ISet::new(span, var, *arg)), known.clone())?;
                            result.push(expr);
                        }
                        pattern if is_last => {
                            // Need to explicitly return match 'value', make safe for efficiency
                            let (la0, mut lps) = force_safe(self.context_mut(), *arg);
                            let mut la = la0.clone();
                            la.mark_compiler_generated();
                            let mc = IClause {
                                span,
                                annotations: annotations.clone(),
                                patterns: vec![pattern],
                                guards: vec![],
                                body: vec![la],
                            };
                            lps.push(IExpr::Case(ICase {
                                span,
                                annotations,
                                args: vec![la0],
                                clauses: vec![mc],
                                fail,
                            }));
                            let (mut exprs, _) = self.uexprs(lps, known.clone())?;
                            result.append(&mut exprs);
                        }
                        pattern => {
                            let body = iter.collect::<Vec<_>>();
                            let mc = IClause {
                                span,
                                annotations: annotations.clone(),
                                patterns: vec![pattern],
                                guards: vec![],
                                body,
                            };
                            let case = IExpr::Case(ICase {
                                span,
                                annotations,
                                args: vec![*arg],
                                clauses: vec![mc],
                                fail,
                            });
                            let (mut exprs, _) = self.uexprs(vec![case], known.clone())?;
                            result.append(&mut exprs);
                            return Ok((result, known));
                        }
                    }
                }
                IExpr::Set(set) => {
                    // Since the set of known variables can grow quite large, try to
                    // minimize the number of union operations on it.
                    let expr = if uexpr_need_known(set.arg.as_ref()) {
                        let known1 = known.union(&new);
                        let le1 = self.uexpr(IExpr::Set(set), known1.clone())?;
                        new = le1.new_vars().clone();
                        le1
                    } else {
                        // We don't need the set of known variables when processing arg0,
                        // so we can postpone the call to union. This will save time
                        // for functions with a huge number of variables.
                        let arg = self.uexpr(*set.arg, Known::default())?;
                        let arg_new = arg.new_vars().insert(set.var.name);
                        let arg_used = arg.used_vars().remove(&set.var.name);
                        new = sets::union(new, arg_new.clone());
                        let mut annotations = set.annotations;
                        annotations.insert_mut(symbols::New, arg_new);
                        annotations.insert_mut(symbols::Used, arg_used);
                        IExpr::Set(ISet {
                            span: set.span,
                            annotations,
                            var: set.var,
                            arg: Box::new(arg),
                        })
                    };
                    match iter.peek() {
                        Some(IExpr::Set(_)) => {
                            result.push(expr);
                        }
                        _ => {
                            result.push(expr);
                            known = known.union(&new);
                            new = rbt_set![];
                        }
                    }
                }
                expr => {
                    let expr = self.uexpr(expr, known.clone())?;
                    let new = expr.new_vars();
                    known = known.union(&new);
                    result.push(expr);
                }
            }
        }
        Ok((result, known))
    }

    fn uexpr(&mut self, expr: IExpr, known: Known) -> anyhow::Result<IExpr> {
        match expr {
            IExpr::Set(ISet {
                span,
                mut annotations,
                var,
                arg,
            }) => {
                let arg = self.uexpr(*arg, known.clone())?;
                let ns = arg
                    .annotations()
                    .new_vars()
                    .unwrap_or_default()
                    .insert(var.name);
                let us = arg
                    .annotations()
                    .used_vars()
                    .unwrap_or_default()
                    .remove(&var.name);
                annotations.insert_mut(symbols::New, ns);
                annotations.insert_mut(symbols::Used, us);
                Ok(IExpr::Set(ISet {
                    span,
                    annotations,
                    var,
                    arg: Box::new(arg),
                }))
            }
            IExpr::LetRec(ILetRec {
                span,
                mut annotations,
                mut defs,
                body,
            }) => {
                let defs = defs
                    .drain(..)
                    .map(|(name, expr)| self.uexpr(expr, known.clone()).map(|e| (name, e)))
                    .try_collect::<Vec<_>>()?;
                let (b1, _) = self.uexprs(body, known)?;
                let used = sets::used_in_any(defs.iter().map(|(_, expr)| expr).chain(b1.iter()));
                annotations.insert_mut(symbols::New, RedBlackTreeSet::new());
                annotations.insert_mut(symbols::Used, used);
                Ok(IExpr::LetRec(ILetRec {
                    span,
                    annotations,
                    defs,
                    body: b1,
                }))
            }
            IExpr::Case(ICase {
                span,
                mut annotations,
                args,
                clauses,
                fail,
            }) => {
                // args will never generate new variables.
                let args = self.uexpr_list(args, known.clone())?;
                let clauses = self.uclauses(clauses, known.clone())?;
                let fail = self.uclause(*fail, known.clone())?;
                let used = sets::union(
                    sets::used_in_any(args.iter()),
                    sets::used_in_any(clauses.iter()),
                );
                let new = if annotations.contains(symbols::ListComprehension) {
                    rbt_set![]
                } else {
                    sets::new_in_all(clauses.iter())
                };
                annotations.insert_mut(symbols::Used, used);
                annotations.insert_mut(symbols::New, new);
                Ok(IExpr::Case(ICase {
                    span,
                    annotations,
                    args,
                    clauses,
                    fail: Box::new(fail),
                }))
            }
            IExpr::If(IIf {
                span,
                mut annotations,
                guards,
                then_body,
                else_body,
            }) => {
                // guards will never generate new variables.
                let guards = self.uexpr_list(guards, known.clone())?;
                let (then_body, _) = self.uexprs(then_body, known.clone())?;
                let (else_body, _) = self.uexprs(else_body, known.clone())?;

                let used = sets::union(
                    sets::used_in_any(guards.iter()),
                    sets::union(
                        sets::used_in_any(then_body.iter()),
                        sets::used_in_any(else_body.iter()),
                    ),
                );
                let new = sets::intersection(
                    sets::new_in_any(then_body.iter()),
                    sets::new_in_any(else_body.iter()),
                );

                annotations.insert_mut(symbols::Used, used);
                annotations.insert_mut(symbols::New, new);
                Ok(IExpr::If(IIf {
                    span,
                    annotations,
                    guards,
                    then_body,
                    else_body,
                }))
            }
            IExpr::Fun(fun) => {
                let clauses = if !known.is_empty() {
                    self.rename_shadowing_clauses(fun.clauses, known.clone())
                } else {
                    fun.clauses
                };
                let span = fun.span;
                let mut annotations = fun.annotations;
                let vars = fun.vars;
                let id = fun.id;
                let name = fun.name;
                let avs = vars.iter().fold(rbt_set![], |mut vs, v| {
                    vs.insert_mut(v.name);
                    vs
                });
                let known1 = if let Some(name) = name.as_ref() {
                    if avs.contains(name) {
                        known.clone()
                    } else {
                        let ks = rbt_set![*name];
                        known.union(&ks)
                    }
                } else {
                    known.clone()
                };
                let known2 = known1.union(&avs);
                let known_in_fun = known2.known_in_fun(name);
                let clauses = self.ufun_clauses(clauses, known_in_fun.clone())?;
                let fail = self.ufun_clause(*fun.fail, known_in_fun)?;
                let used_in_clauses = sets::used_in_any(clauses.iter());
                let used = sets::intersection(used_in_clauses, known1.get().clone());
                let used = sets::subtract(used, avs.clone());
                annotations.insert_mut(symbols::Used, used);
                annotations.insert_mut(symbols::New, rbt_set![]);
                Ok(IExpr::Fun(IFun {
                    span,
                    annotations,
                    id,
                    name,
                    vars,
                    clauses,
                    fail: Box::new(fail),
                }))
            }
            IExpr::Try(texpr) => {
                // No variables are exported from try/catch
                // As of OTP 24, variables bound in the argument are exported
                // to the body clauses, but not the catch or after clauses
                let span = texpr.span;
                let mut annotations = texpr.annotations;
                let (args, _) = self.uexprs(texpr.args, known.clone())?;
                let args_new = sets::new_in_any(args.iter());
                let args_known = known.union(&args_new);
                let (body, _) = self.uexprs(texpr.body, args_known)?;
                let handler = self.uexpr(*texpr.handler, known.clone())?;
                let used = sets::used_in_any(
                    body.iter()
                        .chain(core::iter::once(&handler))
                        .chain(args.iter()),
                );
                let used = sets::intersection(used, known.get().clone());
                annotations.insert_mut(symbols::Used, used);
                annotations.insert_mut(symbols::New, rbt_set![]);
                Ok(IExpr::Try(ITry {
                    span,
                    annotations,
                    vars: texpr.vars,
                    evars: texpr.evars,
                    args,
                    body,
                    handler: Box::new(handler),
                }))
            }
            IExpr::Catch(ICatch {
                span,
                mut annotations,
                body,
            }) => {
                let (body, _) = self.uexprs(body, known)?;
                annotations.insert_mut(symbols::Used, sets::used_in_any(body.iter()));
                Ok(IExpr::Catch(ICatch {
                    span,
                    annotations,
                    body,
                }))
            }
            IExpr::Receive1(recv1) => {
                let span = recv1.span;
                let mut annotations = recv1.annotations;
                let clauses = self.uclauses(recv1.clauses, known)?;
                annotations.insert_mut(symbols::Used, sets::used_in_any(clauses.iter()));
                annotations.insert_mut(symbols::New, sets::new_in_all(clauses.iter()));
                Ok(IExpr::Receive1(IReceive1 {
                    span,
                    annotations,
                    clauses,
                }))
            }
            IExpr::Receive2(recv2) => {
                // Timeout will never generate new variables
                let span = recv2.span;
                let mut annotations = recv2.annotations;
                let timeout = self.uexpr(*recv2.timeout, known.clone())?;
                let clauses = self.uclauses(recv2.clauses, known.clone())?;
                let (action, _) = self.uexprs(recv2.action, known.clone())?;
                let used = sets::union(
                    sets::used_in_any(clauses.iter()),
                    sets::union(sets::used_in_any(action.iter()), timeout.used_vars()),
                );
                let new = if clauses.is_empty() {
                    sets::new_in_any(action.iter())
                } else {
                    sets::intersection(
                        sets::new_in_all(clauses.iter()),
                        sets::new_in_any(action.iter()),
                    )
                };
                annotations.insert_mut(symbols::Used, used);
                annotations.insert_mut(symbols::New, new);
                Ok(IExpr::Receive2(IReceive2 {
                    span,
                    annotations,
                    clauses,
                    timeout: Box::new(timeout),
                    action,
                }))
            }
            IExpr::Protect(IProtect {
                span,
                mut annotations,
                body,
            }) => {
                let (body, _) = self.uexprs(body, known.clone())?;
                let used = sets::used_in_any(body.iter());
                annotations.insert_mut(symbols::Used, used);
                Ok(IExpr::Protect(IProtect {
                    span,
                    annotations,
                    body,
                }))
            }
            IExpr::Binary(mut bin) => {
                let used = bitstr_vars(bin.segments.as_slice(), rbt_set![]);
                bin.annotations_mut().insert_mut(symbols::Used, used);
                Ok(IExpr::Binary(bin))
            }
            IExpr::Apply(mut apply) => {
                let used = lit_list_vars(
                    apply.callee.as_slice(),
                    lit_list_vars(apply.args.as_slice(), rbt_set![]),
                );
                apply.annotations_mut().insert_mut(symbols::Used, used);
                Ok(IExpr::Apply(apply))
            }
            IExpr::PrimOp(mut op) => {
                let used = lit_list_vars(op.args.as_slice(), rbt_set![]);
                op.annotations_mut().insert_mut(symbols::Used, used);
                Ok(IExpr::PrimOp(op))
            }
            IExpr::Call(mut call) => {
                let used = lit_vars(
                    call.module.as_ref(),
                    lit_vars(
                        call.function.as_ref(),
                        lit_list_vars(call.args.as_slice(), rbt_set![]),
                    ),
                );
                call.annotations_mut().insert_mut(symbols::Used, used);
                Ok(IExpr::Call(call))
            }
            mut lit @ IExpr::Literal(_) => {
                lit.annotations_mut().insert_mut(symbols::Used, rbt_set![]);
                Ok(lit)
            }
            mut expr => {
                assert!(expr.is_simple(), "expected simple, got {:#?}", &expr);
                let vars = lit_vars(&expr, RedBlackTreeSet::new());
                expr.annotations_mut().insert_mut(symbols::Used, vars);
                Ok(IExpr::Simple(ISimple::new(expr)))
            }
        }
    }

    fn uexpr_list(&mut self, mut exprs: Vec<IExpr>, known: Known) -> anyhow::Result<Vec<IExpr>> {
        exprs
            .drain(..)
            .map(|expr| self.uexpr(expr, known.clone()))
            .try_collect()
    }

    fn upattern_list(
        &mut self,
        mut patterns: Vec<IExpr>,
        known: Known,
    ) -> anyhow::Result<(
        Vec<IExpr>,
        Vec<IExpr>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
    )> {
        if patterns.is_empty() {
            return Ok((vec![], vec![], rbt_set![], rbt_set![]));
        }

        let pattern = patterns.remove(0);
        let (p1, mut pg, pv, pu) = self.upattern(pattern, known.clone())?;
        let (mut ps1, mut psg, psv, psu) = self.upattern_list(patterns, known.union(&pv))?;
        ps1.insert(0, p1);
        pg.append(&mut psg);
        let vars = sets::union(pv, psv);
        let used = sets::union(pu, psu);
        Ok((ps1, pg, vars, used))
    }

    fn upattern(
        &mut self,
        pattern: IExpr,
        known: Known,
    ) -> anyhow::Result<(
        IExpr,
        Vec<IExpr>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
    )> {
        match pattern {
            IExpr::Var(var) if var.is_wildcard() => {
                let name = self.context_mut().next_var_name(Some(var.span()));
                Ok((
                    IExpr::Var(Var::new(name)),
                    vec![],
                    rbt_set![name],
                    rbt_set![],
                ))
            }
            IExpr::Var(var) => {
                let name = var.name;
                if known.contains(&name) {
                    let new = self.context_mut().next_var(Some(var.span()));
                    let n = new.name;
                    let mut call = ICall::new(
                        var.span(),
                        symbols::Erlang,
                        symbols::EqualStrict,
                        vec![IExpr::Var(new.clone()), IExpr::Var(var.clone())],
                    );
                    {
                        let annos = call.annotations_mut();
                        if annos.contains(symbols::Used) {
                            if let Annotation::Vars(used) = annos.get_mut(symbols::Used).unwrap() {
                                used.insert_mut(name);
                            }
                        } else {
                            let used = rbt_set![name];
                            annos.insert_mut(symbols::Used, used);
                        }
                    }
                    let test = IExpr::Call(call);
                    Ok((IExpr::Var(new), vec![test], rbt_set![n], rbt_set![]))
                } else {
                    Ok((IExpr::Var(var), vec![], rbt_set![name], rbt_set![]))
                }
            }
            IExpr::Cons(ICons {
                span,
                annotations,
                head: h0,
                tail: t0,
            }) => {
                let (h1, mut hg, hv, hu) = self.upattern(*h0, known.clone())?;
                let (t1, mut tg, tv, tu) = self.upattern(*t0, known.union(&hv))?;
                let cons = IExpr::Cons(ICons {
                    span,
                    annotations,
                    head: Box::new(h1),
                    tail: Box::new(t1),
                });
                hg.append(&mut tg);
                Ok((cons, hg, sets::union(hv, tv), sets::union(hu, tu)))
            }
            IExpr::Tuple(ITuple {
                span,
                annotations,
                elements,
            }) => {
                let (elements, esg, esv, eus) = self.upattern_list(elements, known)?;
                Ok((
                    IExpr::Tuple(ITuple {
                        span,
                        annotations,
                        elements,
                    }),
                    esg,
                    esv,
                    eus,
                ))
            }
            IExpr::Map(IMap {
                span,
                annotations,
                arg,
                pairs,
                is_pattern,
            }) => {
                let (pairs, esg, esv, eus) = self.upattern_map(pairs, known)?;
                Ok((
                    IExpr::Map(IMap {
                        span,
                        annotations,
                        arg,
                        pairs,
                        is_pattern,
                    }),
                    esg,
                    esv,
                    eus,
                ))
            }
            IExpr::Binary(IBinary {
                span,
                annotations,
                segments,
            }) => {
                let (segments, esg, esv, eus) = self.upattern_bin(segments, known)?;
                Ok((
                    IExpr::Binary(IBinary {
                        span,
                        annotations,
                        segments,
                    }),
                    esg,
                    esv,
                    eus,
                ))
            }
            IExpr::Alias(IAlias {
                span,
                annotations,
                var: v0,
                pattern: p0,
            }) => {
                let (IExpr::Var(v1), mut vg, vv, vu) = self.upattern(IExpr::Var(v0), known.clone())? else { panic!("expected var") };
                let (p1, mut pg, pv, pu) = self.upattern(*p0, known.union(&vv))?;
                vg.append(&mut pg);
                Ok((
                    IExpr::Alias(IAlias {
                        span,
                        annotations,
                        var: v1,
                        pattern: Box::new(p1),
                    }),
                    vg,
                    sets::union(vv, pv),
                    sets::union(vu, pu),
                ))
            }
            other => Ok((other, vec![], rbt_set![], rbt_set![])),
        }
    }

    fn upattern_map(
        &mut self,
        mut pairs: Vec<IMapPair>,
        known: Known,
    ) -> anyhow::Result<(
        Vec<IMapPair>,
        Vec<IExpr>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
    )> {
        let mut out = Vec::with_capacity(pairs.len());
        let mut tests = vec![];
        let mut pv = rbt_set![];
        let mut pu = rbt_set![];
        for IMapPair { op, key, box value } in pairs.drain(..) {
            assert_eq!(op, MapOp::Exact);
            let (value, mut vg, vn, vu) = self.upattern(value, known.clone())?;
            let (key, _) = self.uexprs(key, known.clone())?;
            let used = used_in_expr(key.as_slice());
            out.push(IMapPair {
                op,
                key,
                value: Box::new(value),
            });
            tests.append(&mut vg);
            pv = sets::union(pv.clone(), vn);
            pu = sets::union(pu.clone(), sets::union(used, vu));
        }
        Ok((out, tests, pv, pu))
    }

    fn upattern_bin(
        &mut self,
        mut segments: Vec<IBitstring>,
        mut known: Known,
    ) -> anyhow::Result<(
        Vec<IBitstring>,
        Vec<IExpr>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
    )> {
        let mut bs = Vec::new();
        let mut out = Vec::with_capacity(segments.len());
        let mut guard = Vec::new();
        let mut vars = RedBlackTreeSet::new();
        let mut used = RedBlackTreeSet::new();
        for segment in segments.drain(..) {
            let (p1, mut pg, pv, pu, bs1) =
                self.upattern_element(segment, known.clone(), bs.clone())?;
            bs = bs1;
            out.push(p1);
            known = known.union(&pv);
            guard.append(&mut pg);
            vars = sets::union(pv, vars.clone());
            used = sets::union(pu, used.clone());
        }
        // In a clause such as <<Sz:8,V:Sz>> in a function head, Sz will both
        // be new and used; a situation that is not handled properly by
        // uclause/4.  (Basically, since Sz occurs in two sets that are
        // subtracted from each other, Sz will not be added to the list of
        // known variables and will seem to be new the next time it is
        // used in a match.)
        //   Since the variable Sz really is new (it does not use a
        // value bound prior to the binary matching), Sz should only be
        // included in the set of new variables. Thus we should take it
        // out of the set of used variables.
        let us = sets::intersection(vars.clone(), used.clone());
        let used = sets::subtract(used, us);
        Ok((out, guard, vars, used))
    }

    fn upattern_element(
        &mut self,
        segment: IBitstring,
        known: Known,
        mut bindings: Vec<(Ident, Ident)>,
    ) -> anyhow::Result<(
        IBitstring,
        Vec<IExpr>,
        RedBlackTreeSet<Ident>,
        RedBlackTreeSet<Ident>,
        Vec<(Ident, Ident)>,
    )> {
        let span = segment.span;
        let annotations = segment.annotations;
        let spec = segment.spec;
        let h0 = *segment.value;
        let mut sz0 = segment.size;
        let (h1, hg, hv, _) = self.upattern(h0.clone(), known.clone())?;
        let bs1 = match &h0 {
            IExpr::Var(v1) => match &h1 {
                IExpr::Var(v2) if v1.name == v2.name => bindings.clone(),
                IExpr::Var(v2) => {
                    let mut bs = bindings.clone();
                    bs.push((v1.name, v2.name));
                    bs
                }
                _ => bindings.clone(),
            },
            _ => bindings.clone(),
        };
        match sz0.pop() {
            Some(IExpr::Var(v)) => {
                let (sz1, used) = rename_bitstr_size(v, bindings);
                let (sz2, _) = self.uexprs(vec![sz1], known)?;
                Ok((
                    IBitstring {
                        span,
                        annotations,
                        value: Box::new(h1),
                        size: sz2,
                        spec,
                    },
                    hg,
                    hv,
                    used,
                    bs1,
                ))
            }
            Some(expr @ IExpr::Literal(_)) => {
                sz0.push(expr);
                let (sz1, _) = self.uexprs(sz0, known)?;
                let used = rbt_set![];
                Ok((
                    IBitstring {
                        span,
                        annotations,
                        value: Box::new(h1),
                        size: sz1,
                        spec,
                    },
                    hg,
                    hv,
                    used,
                    bs1,
                ))
            }
            Some(sz) => {
                let mut sz1 = bindings
                    .drain(..)
                    .map(|(old, new)| {
                        IExpr::Set(ISet::new(
                            new.span(),
                            Var::new(old),
                            IExpr::Var(Var::new(new)),
                        ))
                    })
                    .collect::<Vec<_>>();
                sz1.push(sz);
                sz1.append(&mut sz0);
                let (sz2, _) = self.uexprs(sz1, known)?;
                let used = used_in_expr(sz2.as_slice());
                Ok((
                    IBitstring {
                        span,
                        annotations,
                        value: Box::new(h1),
                        size: sz2,
                        spec,
                    },
                    hg,
                    hv,
                    used,
                    bs1,
                ))
            }
            None => {
                let sz1 = bindings
                    .drain(..)
                    .map(|(old, new)| {
                        IExpr::Set(ISet::new(
                            new.span(),
                            Var::new(old),
                            IExpr::Var(Var::new(new)),
                        ))
                    })
                    .collect::<Vec<_>>();
                let (sz2, _) = self.uexprs(sz1, known)?;
                let used = used_in_expr(sz2.as_slice());
                Ok((
                    IBitstring {
                        span,
                        annotations,
                        value: Box::new(h1),
                        size: sz2,
                        spec,
                    },
                    hg,
                    hv,
                    used,
                    bs1,
                ))
            }
        }
    }

    /// Rename shadowing variables in fun heads.
    ///
    /// Pattern variables in fun heads always shadow variables bound in
    /// the enclosing environment. Because that is the way that variables
    /// behave in Core Erlang, there was previously no need to rename
    /// the variables.
    ///
    /// However, to support splitting of patterns and/or pattern matching
    /// compilation in Core Erlang, there is a need to rename all
    /// shadowing variables to avoid changing the semantics of the Erlang
    /// program.
    ///
    fn rename_shadowing_clauses(
        &mut self,
        mut clauses: Vec<IClause>,
        known: Known,
    ) -> Vec<IClause> {
        clauses
            .drain(..)
            .map(|clause| self.rename_shadowing_clause(clause, known.clone()))
            .collect()
    }

    fn rename_shadowing_clause(&mut self, mut clause: IClause, known: Known) -> IClause {
        let mut isub = vec![];
        let mut osub = vec![];
        let patterns = self.rename_patterns(clause.patterns, known, &mut isub, &mut osub);
        let mut guards = Vec::with_capacity(clause.guards.len());
        if !clause.guards.is_empty() {
            guards.extend(osub.iter().cloned());
            guards.append(&mut clause.guards);
        }
        osub.append(&mut clause.body);
        IClause {
            span: clause.span,
            annotations: clause.annotations,
            patterns,
            guards,
            body: osub,
        }
    }

    fn rename_patterns(
        &mut self,
        mut patterns: Vec<IExpr>,
        known: Known,
        isub: &mut Vec<IExpr>,
        osub: &mut Vec<IExpr>,
    ) -> Vec<IExpr> {
        patterns
            .drain(..)
            .map(|pattern| self.rename_pattern(pattern, known.clone(), isub, osub))
            .collect()
    }

    fn rename_pattern(
        &mut self,
        pattern: IExpr,
        known: Known,
        isub: &mut Vec<IExpr>,
        osub: &mut Vec<IExpr>,
    ) -> IExpr {
        match pattern {
            IExpr::Var(var) if var.is_wildcard() => IExpr::Var(var),
            IExpr::Var(var) => {
                if known.contains(&var.name) {
                    match rename_is_subst(var.name, osub) {
                        Some(new) => new,
                        None => {
                            let span = var.span();
                            let new = self.context_mut().next_var(Some(span));
                            osub.push(IExpr::Set(ISet::new(span, var, IExpr::Var(new.clone()))));
                            IExpr::Var(new)
                        }
                    }
                } else {
                    IExpr::Var(var)
                }
            }
            lit @ IExpr::Literal(_) => lit,
            IExpr::Alias(IAlias {
                span,
                annotations,
                var,
                pattern,
            }) => {
                let IExpr::Var(var) = self.rename_pattern(IExpr::Var(var), known.clone(), isub, osub) else { panic!("expected var") };
                let pattern = self.rename_pattern(*pattern, known, isub, osub);
                IExpr::Alias(IAlias {
                    span,
                    annotations,
                    var,
                    pattern: Box::new(pattern),
                })
            }
            IExpr::Map(IMap {
                span,
                annotations,
                arg,
                pairs,
                is_pattern,
            }) => {
                let pairs = self.rename_pattern_map(pairs, known, isub, osub);
                IExpr::Map(IMap {
                    span,
                    annotations,
                    arg,
                    pairs,
                    is_pattern,
                })
            }
            IExpr::Binary(IBinary {
                span,
                annotations,
                segments,
            }) => {
                let segments = self.rename_pattern_bin(segments, known, isub, osub);
                IExpr::Binary(IBinary {
                    span,
                    annotations,
                    segments,
                })
            }
            IExpr::Cons(ICons {
                span,
                annotations,
                head,
                tail,
            }) => {
                let head = self.rename_pattern(*head, known.clone(), isub, osub);
                let tail = self.rename_pattern(*tail, known, isub, osub);
                IExpr::Cons(ICons {
                    span,
                    annotations,
                    head: Box::new(head),
                    tail: Box::new(tail),
                })
            }
            IExpr::Tuple(ITuple {
                span,
                annotations,
                elements,
            }) => {
                let elements = self.rename_patterns(elements, known, isub, osub);
                IExpr::Tuple(ITuple {
                    span,
                    annotations,
                    elements,
                })
            }
            other => panic!("invalid expression in pattern context {:?}", &other),
        }
    }

    fn rename_pattern_bin(
        &mut self,
        mut segments: Vec<IBitstring>,
        known: Known,
        isub: &mut Vec<IExpr>,
        osub: &mut Vec<IExpr>,
    ) -> Vec<IBitstring> {
        let mut out = Vec::with_capacity(segments.len());
        for IBitstring {
            span,
            annotations,
            size: mut size0,
            value,
            spec,
        } in segments.drain(..)
        {
            let size = rename_get_subst(size0.clone(), isub);
            let value = self.rename_pattern(*value, known.clone(), isub, osub);
            if let Some(IExpr::Var(v)) = size0.pop() {
                assert!(size0.is_empty());
                isub.push(IExpr::Set(ISet::new(span, v, value.clone())));
            }
            out.push(IBitstring {
                span,
                annotations,
                value: Box::new(value),
                size,
                spec,
            });
        }
        out
    }

    fn rename_pattern_map(
        &mut self,
        mut pairs: Vec<IMapPair>,
        known: Known,
        isub: &mut Vec<IExpr>,
        osub: &mut Vec<IExpr>,
    ) -> Vec<IMapPair> {
        pairs
            .drain(..)
            .map(|IMapPair { op, key, value }| {
                let value = self.rename_pattern(*value, known.clone(), isub, osub);
                IMapPair {
                    op,
                    key,
                    value: Box::new(value),
                }
            })
            .collect()
    }
}

fn lit_vars(expr: &IExpr, mut vars: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    match expr {
        IExpr::Cons(ICons {
            ref head, ref tail, ..
        }) => lit_vars(head.as_ref(), lit_vars(tail, vars)),
        IExpr::Tuple(ITuple { ref elements, .. }) => lit_list_vars(elements.as_slice(), vars),
        IExpr::Map(IMap {
            ref arg, ref pairs, ..
        }) => lit_vars(arg.as_ref(), lit_map_vars(pairs.as_slice(), vars)),
        IExpr::Var(Var { name, .. }) => {
            vars.insert_mut(*name);
            vars
        }
        _ => vars,
    }
}

fn lit_list_vars(exprs: &[IExpr], vars: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    exprs.iter().fold(vars, |vs, expr| lit_vars(expr, vs))
}

fn lit_map_vars(pairs: &[IMapPair], vars: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    pairs.iter().fold(vars, |vs, pair| {
        lit_vars(pair.value.as_ref(), lit_list_vars(pair.key.as_slice(), vs))
    })
}

fn bitstr_vars(segments: &[IBitstring], vars: RedBlackTreeSet<Ident>) -> RedBlackTreeSet<Ident> {
    segments.iter().fold(vars, |vs, seg| {
        lit_vars(seg.value.as_ref(), lit_list_vars(seg.size.as_slice(), vs))
    })
}

fn rename_bitstr_size(var: Var, mut bs: Vec<(Ident, Ident)>) -> (IExpr, RedBlackTreeSet<Ident>) {
    let vname = var.name;
    for (v1, name) in bs.drain(..) {
        if vname == v1 {
            return (IExpr::Var(Var::new(name)), rbt_set![name]);
        }
    }
    (IExpr::Var(var), rbt_set![vname])
}

fn rename_get_subst(mut exprs: Vec<IExpr>, sub: &mut Vec<IExpr>) -> Vec<IExpr> {
    match exprs.len() {
        1 => match exprs.pop().unwrap() {
            IExpr::Var(var) => match rename_is_subst(var.name, sub) {
                None => vec![IExpr::Var(var)],
                Some(new) => vec![new],
            },
            lit @ IExpr::Literal(_) => vec![lit],
            expr => {
                let mut result = sub.clone();
                result.push(expr);
                result
            }
        },
        0 => sub.clone(),
        _ => {
            let mut result = sub.clone();
            result.extend(exprs.drain(..));
            result
        }
    }
}

fn rename_is_subst(var: Ident, sub: &mut Vec<IExpr>) -> Option<IExpr> {
    sub.iter().find_map(|expr| match expr {
        IExpr::Set(set) if set.var.name == var => Some(*set.arg.clone()),
        _ => None,
    })
}

fn used_in_expr(exprs: &[IExpr]) -> RedBlackTreeSet<Ident> {
    exprs.iter().rfold(RedBlackTreeSet::new(), |used, expr| {
        sets::union(expr.used_vars(), sets::subtract(used, expr.new_vars()))
    })
}

fn uexpr_need_known(expr: &IExpr) -> bool {
    match expr {
        IExpr::Call(_)
        | IExpr::Apply(_)
        | IExpr::Binary(_)
        | IExpr::PrimOp(_)
        | IExpr::Literal(_) => false,
        expr => !expr.is_simple(),
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
