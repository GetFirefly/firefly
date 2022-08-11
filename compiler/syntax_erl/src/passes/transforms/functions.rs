use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::mem;
use std::rc::Rc;

use liblumen_diagnostics::*;
use liblumen_pass::Pass;
use liblumen_syntax_ssa as syntax_ssa;

use crate::ast::*;
use crate::evaluator;
use crate::visit::ast::{self, VisitMut};

use super::{FunctionContext, NoMatchError};

impl CanonicalizeFunction {
    /// Flatten top-level expressions
    fn exprs(&mut self, exprs: Vec<Expr>) -> Result<Vec<Expr>, ()> {
        let mut results = Vec::with_capacity(exprs.len());
        for expr in exprs.drain(..) {
            let (expr, eps) = self.expr(expr)?;
            if let Some(mut eps) = eps {
                results.append(&mut eps);
            }
            results.push(expr);
        }
        Ok(results)
    }

    /// Canonicalize expressions
    fn expr(&mut self, expr: Expr) -> Result<(Expr, Vec<Expr>), ()> {
        match expr {
            var @ Expr::Var(_) => Ok((var, vec![])),
            lit @ Expr::Literal(_) => Ok((lit, vec![])),
            Expr::Cons(Cons { span, head, tail }) => {
                let (mut safe, pre) = self.safe_list(vec![*head, *tail])?;
                let tail = safe.pop().unwrap();
                let head = safe.pop().unwrap();
                Ok((
                    Expr::Cons(Cons {
                        span,
                        head: Box::new(head),
                        tail: Box::new(tail),
                    }),
                    pre,
                ))
            }
            Expr::ListComprehension(ListComprehension {
                span,
                body,
                qualifiers,
            }) => {
                let qualifiers = self.preprocess_quals(span, qualifiers)?;
                self.lc_tq(span, *body, qualifiers)
            }
            Expr::BinaryComprehension(BinaryComprehension {
                span,
                body,
                qualifiers,
            }) => self.bc_tq(span, *body, qualifiers),
            Expr::Tuple(Tuple { span, elements }) => {
                let (elements, pre) = self.safe_list(elements);
                Ok((Expr::Tuple(Tuple { span, elements }), pre))
            }
            Expr::Map(map) => self.map_build_pairs(map),
            Expr::MapUpdate(map) => self.expr_map(map),
            Expr::Binary(bin) => self.expr_bin(bin),
            Expr::Begin(Begin { span, mut body }) => {
                // Inline the block directly
                let last = body.pop().unwrap();
                let mut exprs = self.exprs(block.exprs)?;
                let (last, mut pre) = self.expr(last)?;
                exprs.append(&mut pre);
                Ok((last, exprs))
            }
            Expr::If(If { span, clauses }) => {
                let clauses = self.clauses(clauses)?;
                Ok((Expr::If(If { span, clauses }), vec![]))
            }
            Expr::Case(Case {
                span,
                expr,
                clauses,
            }) => {
                let (expr, eps) = self.novars(*expr);
                let clauses = self.clauses(clauses)?;
                Ok((
                    Expr::Case(Case {
                        span,
                        expr: Box::new(expr),
                        clauses,
                    }),
                    eps,
                ))
            }
            Expr::Receive(Receive {
                span,
                clauses,
                after: None,
            }) => {
                let clauses = self.clauses(clauses.unwrap())?;
                Ok((
                    Expr::Receive(Receive {
                        span,
                        clauses,
                        after: None,
                    }),
                    None,
                ))
            }
            Expr::Receive(Receive {
                span,
                clauses,
                after: Some(after),
            }) => {
                let (timeout, eps) = self.novars(*after.timeout);
                let after_body = self.exprs(after.body)?;
                let clauses = match clauses {
                    None => None,
                    Some(clauses) => Some(self.clauses(clauses)?),
                };
                Ok((
                    Expr::Receive(Receive {
                        span,
                        clauses,
                        after: Some(After {
                            span: after.span,
                            timeout: Box::new(timeout),
                            body: after_body,
                        }),
                    }),
                    eps,
                ))
            }
            // try .. catch .. end
            Expr::Try(Try {
                span,
                exprs,
                clauses: None,
                catch_clauses: Some(ccs),
                after: None,
            }) => {
                let exprs = self.exprs(exprs)?;
                let v = self.context.next_var(Some(span));
                let ccs = self.try_exception(ccs)?;
                let cs = vec![Clause {
                    span,
                    pattern: Expr::Var(Var(v)),
                    guard: None,
                    body: vec![Expr::Var(Var(v))],
                }];
                Ok((
                    Expr::Try(Try {
                        span,
                        exprs,
                        clauses: Some(cs),
                        catch_clauses: Some(ccs),
                        after: None,
                    }),
                    None,
                ))
            }
            // try .. of .. catch .. end
            Expr::Try(Try {
                span,
                exprs,
                clauses: Some(cs),
                catch_clauses: Some(ccs),
                after: None,
            }) => {
                let exprs = self.exprs(exprs)?;
                let v = self.context.next_var(Some(span));
                let cs = self.clauses(cs)?;
                let fpat = self.context.next_var(Some(span));
                let ccs = self.try_exception(ccs)?;
                let cs = vec![Clause {
                    span,
                    pattern: Expr::Var(Var(v)),
                    guard: None,
                    body: vec![Case {
                        span,
                        expr: Box::new(Expr::Var(Var(v))),
                        clauses: cs,
                    }],
                }];
                Ok((
                    Expr::Try(Try {
                        span,
                        exprs,
                        clauses: Some(cs),
                        catch_clauses: Some(ccs),
                        after: None,
                    }),
                    None,
                ))
            }
            // try .. after .. end
            Expr::Try(Try {
                span,
                exprs,
                clauses: None,
                catch_clauses: None,
                after: Some(after),
            }) => self.try_after(span, exprs, after),
            // try .. [of ...] [catch ... ] after .. end
            Expr::Try(Try {
                span,
                exprs,
                clauses,
                catch_clauses,
                after: Some(after),
            }) => self.expr(Expr::Try(Try {
                span,
                exprs: vec![Expr::Try(Try {
                    span,
                    exprs,
                    clauses,
                    catch_clauses,
                    after: None,
                })],
                clauses: None,
                catch_clauses: None,
                after: Some(after),
            })),
            Expr::Catch(Catch { span, expr }) => {
                let (expr, mut pre) = self.expr(*expr)?;
                pre.push(expr);
                Ok((Expr::Catch(Catch { span, body: pre }), None))
            }
            Expr::Fun(fun) => self.fun_tq(fun),
            Expr::Apply(Apply {
                span,
                callee,
                mut args,
            }) => match *callee {
                callee @ Expr::FunctionName(_) => {
                    let (args, mut pre) = self.safe_list(args);
                    Ok((
                        Expr::Apply(Apply {
                            span,
                            callee: Box::new(callee),
                            args,
                        }),
                        pre,
                    ))
                }
                Expr::Remote(Remote {
                    module, function, ..
                }) => {
                    let mut safes = vec![*module, *function];
                    safes.append(&mut args);
                    let (mut safes, pre) = self.safe_list(safes);
                    let module = safes.pop().unwrap();
                    let function = safes.pop().unwrap();
                    let args = safes;
                    Ok((
                        Expr::Apply(Apply {
                            span,
                            callee: Box::new(Expr::Remote(Remote {
                                span: rspan,
                                module: Box::new(module),
                                function: Box::new(function),
                            })),
                            args,
                        }),
                        pre,
                    ))
                }
                Expr::Literal(Literal::Atom(f)) => {
                    let (args, pre) = self.safe_list(args);
                    let callee =
                        Expr::FunctionName(FunctionName::Unresolved(UnresolvedFunctionName {
                            span: f.span,
                            module: None,
                            function: Name::Atom(f),
                            arity: args.len(),
                        }));
                    Ok((
                        Expr::Apply(Apply {
                            span,
                            callee: Box::new(callee),
                            args,
                        }),
                        aps,
                    ))
                }
                callee => {
                    let (callee, mut pre) = self.safe(callee);
                    let (args, mut pre2) = self.safe_list(args);
                    pre.append(&mut pre2);
                    Ok((
                        Expr::Apply(Apply {
                            span,
                            callee: Box::new(callee),
                            args,
                        }),
                        pre,
                    ))
                }
            },
            Expr::Match(Match {
                span,
                pattern,
                expr,
            }) => {
                let (pattern, expr) = self.fold_match(expr, pattern);
                // set_wanted(pattern)
                let (expr, pre) = self.novars(expr);
                let pattern = self.pattern(pattern);
                let fpat = self.context.next_var();
                match pattern {
                    Err(NoMatchError) => {
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
                        self.reporter.show_warning(
                            "impossible match",
                            &[(span, "this pattern can never match")],
                        );
                        let (expr, mut pre) = self.safe(expr);
                        let pattern = self.sanitize(pattern);
                        let pattern = self.pattern(pattern).unwrap();
                        pre.push(Expr::PrimOp(PrimOp::match_fail(
                            span,
                            vec![tuple_with_span!(span, atom!(badmatch), expr.clone())],
                        )));
                        Ok((
                            Expr::Match(Match {
                                span,
                                pattern: Box::new(pattern),
                                expr: Box::new(expr),
                            }),
                            pre,
                        ))
                    }
                    Ok(pattern) => {
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
                        let (pattern, expr, mut pre2) = self.letify_aliases(pattern, expr);
                        pre.append(&mut pre2);
                        Ok((
                            Expr::Match(Match {
                                span,
                                pattern: Box::new(pattern),
                                expr: Box::new(expr),
                            }),
                            eps,
                        ))
                    }
                }
            }
            Expr::BinaryExpr(BinaryExpr { span, op, lhs, rhs }) if op == BinaryOp::Append => {
                match *lhs {
                    Expr::ListComprehension(ListComprehension {
                        span,
                        body,
                        qualifiers,
                    }) => {
                        // Optimize '++' due to the list comprehension algorithm
                        //
                        // This avoids quadratic complexity if there is a chain of list comprehensions
                        // without generators being combined with '++', by forcing evaluation of the right-hand
                        // operand now.
                        let (rhs, mut pre) = self.safe(*rhs)?;
                        let qualifiers = self.preprocess_quals(qualifiers)?;
                        let (y, mut pre2) = self.lc_tq(span, *body, qualifiers)?;
                        pre.append(&mut pre2);
                        Ok((y, pre))
                    }
                    lhs => {
                        let (args, pre) = self.safe_list(vec![lhs, *rhs])?;
                        let callee = Expr::FunctionName(FunctionName::new(
                            span,
                            symbols::Erlang,
                            op.to_symbol(),
                            2,
                        ));
                        let expr = Expr::Apply(Apply::new(span, callee, args));
                        Ok((expr, pre))
                    }
                }
            }
            Expr::BinaryExpr(BinaryExpr { span, op, lhs, rhs }) if op == BinaryOp::AndAlso => {
                let v0 = self.context.next_var(Some(span));
                let atom_false = atom!(false);
                let expr = self.make_bool_switch(*lhs, v0, *rhs, atom_false);
                self.expr(expr)
            }
            Expr::BinaryExpr(BinaryExpr { span, op, lhs, rhs }) if op == BinaryOp::OrElse => {
                let v0 = self.context.next_var(Some(span));
                let atom_true = atom!(false);
                let expr = self.make_bool_switch(*lhs, v0, atom_true, *rhs);
                self.expr(expr)
            }
            Expr::BinaryExpr(BinaryExpr { span, op, lhs, rhs }) => {
                let (args, pre) = self.safe_list(vec![*lhs, *rhs])?;
                let callee =
                    Expr::FunctionName(FunctionName::new(span, symbols::Erlang, op.to_symbol(), 2));
                let expr = Expr::Apply(Apply::new(span, callee, args));
                Ok((expr, pre))
            }
            Expr::UnaryExpr(UnaryExpr { span, op, operand }) => {
                let (operand, pre) = self.safe(*operand)?;
                let callee =
                    Expr::FunctionName(FunctionName::new(span, symbols::Erlang, op.to_symbol(), 1));
                let expr = Expr::Apply(Apply::new(span, callee, vec![*operand]));
                Ok((expr, pre))
            }
            _expr => unreachable!(),
        }
    }

    fn make_bool_switch(&mut self, expr: Expr, var: Ident, t: Expr, f: Expr) -> Expr {
        let span = expr.span();
        let badarg = atom!(badarg);
        let atom_true = atom!(true);
        let atom_false = atom!(false);
        let var = Expr::Var(Var(var));
        let error = tuple_with_span!(span, badarg, var.clone());
        let call = apply!(span, erlang, error, (error));
        Expr::Case(Case {
            span,
            expr: Box::new(expr),
            clauses: vec![
                Clause::new(span, atom_true, vec![], vec![t], true),
                Clause::new(span, atom_false, vec![], vec![f], true),
                Clause::new(span, var, vec![], vec![call], true),
            ],
        })
    }

    fn try_exception(&mut self, clauses: Vec<Clause>) -> Result<Vec<Clause>, ()> {
        let span = clauses[0].span;
        // Tag, Value, Info
        let tag = self.context.next_var(Some(span));
        let value = self.context.next_var(Some(span));
        let info = self.context.next_var(Some(span));
        let evs = vec![
            Expr::Var(Var(tag)),
            Expr::Var(Var(value)),
            Expr::Var(Var(info)),
        ];
        let clauses = self.clauses(clauses)?;
        let clauses = self.try_build_stacktrace(clauses, tag);
        let consolidated = Clause {
            span,
            pattern: Expr::Tuple(Tuple {
                span,
                elements: evs.clone(),
            }),
            guard: None,
            body: vec![Expr::Case(Case {
                span,
                expr: Box::new(Expr::Tuple(Tuple {
                    span,
                    elements: evs.clone(),
                })),
                clauses,
            })],
        };
        Ok(vec![consolidated])
    }

    fn try_build_stacktrace(&mut self, clauses: Vec<Clause>, raw_stack: Ident) -> Vec<Clause> {
        let mut results = Vec::with_capacity(clauses.len());
        for mut clause in clauses.drain(..) {
            let Expr::Tuple(tuple) = clause.pattern else { panic!("expected tuple pattern"); };
            let span = tuple.span;
            let stack = tuple.elements[2];
            match stack {
                Expr::Var(v) if v.is_wildcard() => {
                    // Stacktrace variable is not used. Nothing to do.
                    results.push(clause);
                }
                _ => {
                    let class = tuple.elements[0];
                    let exception = tuple.elements[1];
                    let pattern = Expr::Tuple(Tuple {
                        span,
                        elements: vec![class, exception, Expr::Var(Var(raw_stack))],
                    });
                    let build = Expr::PrimOp(PrimOp {
                        span,
                        name: symbols::BuildStacktrace,
                        args: vec![Expr::Var(Var(raw_stack))],
                    });
                    let set = Expr::Let(Let {
                        span,
                        var: Var(raw_stack),
                        expr: Box::new(build),
                    });
                    let mut body = vec![set];
                    body.append(&mut clause.body);
                    let clause = Clause {
                        span: clause.span,
                        pattern,
                        guards: clause.guards,
                        body,
                    };
                    results.push(clause);
                }
            }
        }
        results
    }

    /// Generate an internal safe expression for a list of expressions
    fn safe_list(&mut self, exprs: Vec<Expr>) -> Result<(Vec<Expr>, Vec<Expr>), ()> {
        let mut safes: Vec<Expr> = vec![];
        let mut pres: Vec<Vec<Expr>> = vec![];
        for expr in exprs.drain(..) {
            let (expr, pre) = self.safe(expr);
            if let Some(mut ps) = pres.pop() {
                match ps.pop() {
                    Some(Expr::Nested(mut bodies)) if ps.len() == 0 => {
                        // A cons within a cons
                        safes.push(expr);
                        pres.append(&mut bodies);
                        pres.push(pre);
                    }
                    Some(last) => {
                        safes.push(expr);
                        ps.push(last);
                        pres.push(ps);
                        pres.push(pre);
                    }
                }
            } else {
                safes.push(expr);
                pres.push(pre);
            }
        }

        let pres = pres.drain(..).filter(|ps| ps.len() > 0).collect();
        match pres.len() {
            0 => Ok((safes, vec![])),
            1 => Ok((safes, pres.pop().unwrap())),
            _ => {
                // Two or more bodies, they see the same variables.
                Ok((safes, vec![Expr::Nested(pres)]))
            }
        }
    }

    /// Generate an internal safe expression. These are simples without binaries which can fail
    fn safe(&mut self, expr: Expr) -> Result<(Expr, Vec<Expr>), ()> {
        let (expr, mut pre) = self.expr(expr)?;
        let (safe, mut safe_pre) = self.force_safe(expr);
        pre.append(&mut safe_pre);
        Ok((safe, pre))
    }

    fn force_safe(&mut self, expr: Expr) -> (Expr, Vec<Expr>) {
        match expr {
            Expr::Match(Match {
                span,
                pattern,
                expr,
            }) => {
                let (expr, mut pre) = self.force_safe(*expr);
                // Make sure we don't duplicate the expression `expr`
                match expr {
                    var @ Expr::Var(_) => {
                        // Expression is a variable, so we generate `Pattern = Var, Var.`
                        pre.push(Expr::Match(Match {
                            span,
                            pattern,
                            expr: Box::new(var),
                        }));
                        (expr, pre)
                    }
                    other => {
                        // Expression is not a variable, so we generate `NewVar = Pattern = Var, NewVar`
                        let var = Var(self.context.next_var(Some(span)));
                        pre.push(Expr::Match(Match {
                            span,
                            pattern: Box::new(Expr::Alias(Alias {
                                var: var.clone(),
                                pattern,
                            })),
                            expr: Box::new(other),
                        }));
                        (Expr::Var(var.clone()), pre)
                    }
                }
            }
            expr => {
                if expr.is_safe() {
                    Ok((expr, vec![]))
                } else {
                    let span = expr.span();
                    let var = Var(self.context.next_var(Some(span)));
                    (
                        Expr::Var(var),
                        vec![Expr::Let(Let {
                            span,
                            var,
                            expr: Box::new(expr),
                        })],
                    )
                }
            }
        }
    }

    fn pattern_list(&mut self, params: Vec<Expr>) -> Result<Vec<Expr>, Vec<Expr>> {
        let mut patterns = Vec::with_capacity(params.len());
        let mut nomatch = false;
        for pattern in params.iter().cloned() {
            match self.pattern(pattern) {
                Ok(pattern) => patterns.push(pattern),
                Err(NoMatchError) => {
                    nomatch = true;
                    break;
                }
            }
        }
        if nomatch {
            Err(params)
        } else {
            Ok(patterns)
        }
    }

    fn pattern(&mut self, pattern: Expr) -> Result<Expr, NoMatchError> {
        match pattern {
            Expr::Var(_) | Expr::Literal(_) => Ok(pattern),
            Expr::Cons(Cons {
                span,
                mut head,
                mut tail,
            }) => {
                let h = self.pattern(mem::replace(
                    head.as_mut(),
                    Expr::Literal(Literal::Nil(span)),
                ))?;
                let t = self.pattern(mem::replace(
                    tail.as_mut(),
                    Expr::Literal(Literal::Nil(span)),
                ))?;
                mem::replace(head.as_mut(), h);
                mem::replace(tail.as_mut(), t);
                Ok(Expr::Cons(Cons { span, head, tail }))
            }
            Expr::Tuple(Tuple { span, elements }) => {
                let elements = self.pattern_list(elements).map_err(|_| NoMatchError)?;
                Ok(Expr::Tuple(Tuple { span, elements }))
            }
            Expr::Map(Map { span, fields }) => {
                let fields = self.pattern_map_pairs(fields)?;
                Ok(Expr::Map(Map { span, fields }))
            }
            Expr::Binary(Binary { span, elements }) => {
                let elements = self.pattern_bin(elements)?;
                Ok(Expr::Binary(Binary { span, elements }))
            }
            Expr::Match(Match {
                span,
                pattern,
                expr,
            }) => {
                let pattern = self.pattern(*pattern)?;
                let var = pattern.as_var().unwrap();
                let e = self.pattern(mem::replace(
                    expr.as_mut(),
                    Expr::Literal(Literal::Nil(span)),
                ))?;
                mem::replace(expr.as_mut(), e);
                Ok(Expr::Alias(Alias { var, pattern: expr }))
            }
            // Compile-time expressions
            Expr::BinaryExpr(BinaryExpr { span, op, lhs, rhs }) if op == BinaryOp::Append => {
                match lhs {
                    Expr::Literal(Literal::Nil(_)) => self.pattern(*rhs),
                    Expr::Cons(Cons {
                        span: cons_span,
                        head,
                        tail,
                    }) => {
                        // Recursively expand [H | (T ++ Rhs)]
                        let tail = self.pattern(Expr::BinaryExpr(BinaryExpr {
                            span,
                            op,
                            rhs,
                            lhs: tail,
                        }))?;
                        Ok(Expr::Cons(Cons {
                            span: cons_span,
                            head,
                            tail: Box::new(tail),
                        }))
                    }
                    Expr::Literal(Literal::String(s)) => {
                        let cspan = s.span;
                        let mut tail = Some(Expr::Literal(Literal::Nil(cspan)));
                        for head in s.as_str().get().chars().rev().map(|c| {
                            Expr::Literal(Literal::Integer(cspan, Integer::Small(c as i64)))
                        }) {
                            let t = tail.take().unwrap();
                            tail = Some(Expr::Cons(Cons {
                                span: cspan,
                                head: Box::new(head),
                                tail: Box::new(t),
                            }));
                        }
                        self.pattern(Expr::BinaryExpr(BinaryExpr {
                            span,
                            op,
                            lhs: Box::new(tail.unwrap()),
                            rhs,
                        }))
                    }
                    expr => {
                        let expr_span = expr.span();
                        match evaluator::eval_expr(&expr, None) {
                            Ok(t) => {
                                let lhs = t.to_expr(expr_span);
                                let rhs = self.pattern(*rhs)?;
                                Ok(Expr::BinaryExpr(BinaryExpr {
                                    span,
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                }))
                            }
                            Err(err) => {
                                self.reporter.error(err);
                                Err(NoMatchError)
                            }
                        }
                    }
                }
            }
            expr @ Expr::BinaryExpr(_) => {
                let expr_span = expr.span();
                match evaluator::eval_expr(&expr, None) {
                    Ok(value) => Ok(value.to_expr(expr_span)),
                    Err(_) => Ok(expr),
                }
            }
            expr => {
                self.show_error(
                    "illegal pattern",
                    &[(
                        expr.span(),
                        "this expression is not allowed in a pattern context",
                    )],
                );
                Err(NoMatchError)
            }
        }
    }

    fn pattern_bin(
        &mut self,
        elements: Vec<BinaryElement>,
    ) -> Result<Vec<BinaryElement>, NoMatchError> {
        let elements = self.pattern_bin_expand_strings(elements);
        self.pattern_bin_segments(elements)
    }

    fn pattern_bin_expand_strings(&mut self, elements: Vec<BinaryElement>) -> Vec<BinaryElement> {
        elements.rfold(
            vec![],
            |BinaryElement {
                 span,
                 bit_expr,
                 bit_size,
                 specifier,
             },
             mut acc| {
                match bit_expr {
                    // <<"foo">>
                    Expr::Literal(Literal::String(s))
                        if bit_size.is_none() && specifier.is_none() =>
                    {
                        let chars = s.as_str().get().chars().rev().map(|c| c as i64).collect();
                        self.bin_expand_string(chars, span, 0, 0, acc)
                    }
                    Expr::Literal(Literal::String(s)) => {
                        let cspan = s.span;
                        for c in s.as_str().get().chars().rev() {
                            acc.push(BinaryElement {
                                span,
                                bit_expr: Box::new(Expr::Literal(Literal::Char(cspan, c))),
                                bit_size: bit_size.clone(),
                                specifier: specifier.clone(),
                            });
                        }
                        acc
                    }
                    bit_expr => {
                        acc.push(BinaryElement {
                            span,
                            bit_expr,
                            bit_size,
                            specifier,
                        });
                        acc
                    }
                }
            },
        )
    }

    fn pattern_bin_segments(
        &mut self,
        elements: Vec<BinaryElement>,
    ) -> Result<Vec<BinaryElement>, NoMatchError> {
        let mut segments = Vec::with_capacity(elements.len());
        for element in elements {
            segments.push(self.pattern_bin_segment(element)?);
        }
        Ok(segments)
    }

    fn pattern_bin_segment(
        &mut self,
        element: BinaryElement,
    ) -> Result<BinaryElement, NoMatchError> {
        let (size, spec) = self.make_bit_type(element.span, element.bit_size, element.specifier)?;
        let pattern_value = self.pattern(element.bit_expr)?;
        let pattern_value = if spec.is_float() {
            pattern_value.coerce_to_float()
        } else {
            pattern_value
        };
        let pattern_size = self.expr(evaluator::eval_expr(size, None).unwrap());
        Ok(BinaryElement {
            span: element.span,
            bit_expr: pattern_value,
            bit_size: Some(pattern_size),
            specifier: Some(spec),
        })
    }

    fn make_bit_type(
        &mut self,
        span: SourceSpan,
        size: Option<Expr>,
        spec: Option<BinaryEntrySpecifier>,
    ) -> Result<(Option<Expr>, BinaryEntrySpecifier), NoMatchError> {
        match size {
            Some(Expr::Literal(Literal::Atom(a))) if a.name == symbols::All => {
                self.reporter
                    .show_error("invalid bit pattern", &[(span, "invalid size")]);
                Err(NoMatchError)
            }
            size => Ok(self.apply_bit_defaults(span, size, spec.unwrap_or_default())),
        }
    }

    fn apply_bit_defaults(
        &mut self,
        span: SourceSpan,
        size: Option<Expr>,
        spec: BinaryEntrySpecifier,
    ) -> (Option<Expr>, BinaryEntrySpecifier) {
        match spec {
            BinaryEntrySpecifier::Binary { .. } if size.is_none() => {
                (
                    Some(Expr::Literal(Literal::Atom(Ident::new(symbols::All, span)))),
                    spec,
                )
            }
            BinaryEntrySpecifier::Integer { .. } if size.is_none() => {
                if unit != 1 {
                    self.reporter.show_error(
                        "invalid bit pattern",
                        &[(span, "must specify size with custom unit")],
                    );
                }
                (
                    Some(Expr::Literal(Literal::Integer(span, Integer::Small(8)))),
                    spec,
                )
            }
            BinaryEntrySpecifier::Float { unit, .. } if size.is_none() => {
                if unit != 1 {
                    self.reporter.show_error(
                        "invalid bit pattern",
                        &[(span, "must specify size with custom unit")],
                    );
                }
                (
                    Some(Expr::Literal(Literal::Integer(span, Integer::Small(64)))),
                    spec,
                )
            }
            _ => (size, spec),
        }
    }

    fn bin_expand_string(
        &mut self,
        s: Vec<i64>,
        span: SourceSpan,
        value: i64,
        size: usize,
        mut last: Vec<BinaryElement>,
    ) -> Vec<BinaryElement> {
        const COLLAPSE_MAX_SIZE_SEGMENT: usize = 1024;

        let mut elements = Vec::with_capacity(last.len() + 1);

        if size >= COLLAPSE_MAX_SIZE_SEGMENT {
            elements.push(self.make_combined(span, value, size));
            let mut expanded = self.bin_expand_string(s, span, 0, 0, last);
            elements.append(&mut expanded);
            return elements;
        }

        match s.pop() {
            None => {
                elements.push(self.make_combined(span, value, size));
                elements.append(&mut last);
                elements
            }
            Some(head) => {
                let value = (value << 8) | head;
                let size = size + 8;
                self.bin_expand_string(s, span, value, size, last)
            }
        }
    }

    fn make_combined(span: SourceSpan, val: i64, size: usize) -> BinaryElement {
        BinaryElement {
            span,
            bit_expr: Expr::Literal(Literal::Integer(span, Integer::Small(val))),
            bit_size: Some(Expr::Literal(Literal::Integer(
                span,
                Integer::Small(size as i64),
            ))),
            specifier: Some(BinaryEntrySpecifier::Integer {
                signed: false,
                endianness: Endianness::Big,
                unit: 1,
            }),
        }
    }

    fn pattern_map_pairs(&mut self, fields: Vec<MapField>) -> Result<Vec<MapField>, NoMatchError> {
        let mut patterns = Vec::with_capacity(fields.len());
        for field in fields.drain(..) {
            match field {
                MapField::Exact { span, key, value } => {
                    let key = self.expr(self.partial_eval(key));
                    let pattern = self.pattern(value)?;
                }
                _field => unreachable!(
                    "should not be possible to have map update expressions in a pattern"
                ),
            }
        }
        self.normalize_alias_map_pairs(patterns.drain(..));
    }

    fn letify_aliases(&mut self, pattern: Expr, expr: Expr) -> (Expr, Expr, Vec<Expr>) {
        match pattern {
            Expr::Alias(Alias { var, pattern }) => {
                let (pattern1, expr1, mut pre) = self.letify_aliases(pattern, Expr::Var(var));
                pre.insert(0, Expr::Let(Let { var, expr }));
                (pattern1, expr1, pre)
            }
            pattern => (pattern, expr, vec![]),
        }
    }

    // Fold nested matches into one match with aliased patterns
    fn fold_match(&mut self, expr0: Expr, pattern0: Expr) -> (Expr, Expr) {
        match expr0 {
            Expr::Match(Match {
                span,
                expr: box expr1,
                pattern: box pattern1,
            }) => (
                Expr::Match(Match {
                    span,
                    pattern: pattern0,
                    expr: pattern1,
                }),
                expr1,
            ),
            _ => (pattern0, expr0),
        }
    }

    fn normalize_alias_patterns(
        &mut self,
        pattern0: Expr,
        pattern1: Expr,
    ) -> Result<Expr, NoMatchError> {
        match (pattern0, pattern1) {
            (Expr::Var(x), Expr::Var(y)) if x.name == y.name => Ok(Expr::Var(x)),
            (Expr::Var(x), alias @ Expr::Alias(Alias { var: ref y, .. })) if x.name == y.name => {
                Ok(alias)
            }
            (Expr::Var(var), Expr::Alias(Alias { var: y, pattern })) => Ok(Expr::Alias(Alias {
                var: y,
                pattern: Box::new(Expr::Alias(Alias { var, pattern })),
            })),
            (Expr::Var(var), pattern) => Ok(Expr::Alias(Alias {
                var,
                pattern: Box::new(pattern),
            })),
            (alias @ Expr::Alias(Alias { var: x, .. }), Expr::Var(y)) if x.name == y.name => {
                Ok(alias)
            }
            (
                Expr::Alias(Alias {
                    var: v1,
                    pattern: p1,
                }),
                Expr::Alias(Alias {
                    var: v2,
                    pattern: p2,
                }),
            ) => {
                let pattern = self.normalize_alias_patterns(p1, p2)?;
                if v1.name == v2.name {
                    Ok(Expr::Alias(Alias {
                        var: v1,
                        pattern: Box::new(pattern),
                    }))
                } else {
                    self.normalize_alias_patterns(
                        Expr::Var(v1),
                        self.normalize_alias_patterns(Expr::Var(v2), pattern)?,
                    )
                }
            }
            (Expr::Alias(Alias { var, pattern: p1 }), p2) => Ok(Expr::Alias(Alias {
                var,
                pattern: Box::new(self.normalize_alias_patterns(*p1, p2)),
            })),
            (
                Expr::Map(Map {
                    span,
                    fields: fields1,
                }),
                Expr::Map(Map { fields: field2, .. }),
            ) => {
                let fields = self.normalize_alias_map_pairs(fields1.iter().concat(field2.iter()));
                Ok(Expr::Map(Map { span, fields }))
            }
            (pattern, Expr::Var(var)) => Ok(Expr::Alias(Alias {
                var,
                pattern: Box::new(pattern),
            })),
            (p1, Expr::Alias(Alias { var, pattern: p2 })) => Ok(Expr::Alias(Alias {
                var,
                pattern: Box::new(self.normalize_alias_patterns(p1, *p2)?),
            })),
            (p1, p2) => {
                // Aliases between binaries are not allowed, so the only legal patterns that remain are data structures
                match (p1.is_data_constructor(), p2.is_data_constructor()) {
                    (true, false) => {
                        self.reporter.show_error(
                            "illegal pattern",
                            &[(
                                p2.span(),
                                "expected literal, cons or tuple constructor here",
                            )],
                        );
                        Err(NoMatchError)
                    }
                    (false, true) => {
                        self.reporter.show_error(
                            "illegal pattern",
                            &[(
                                p1.span(),
                                "expected literal, cons or tuple constructor here",
                            )],
                        );
                        Err(NoMatchError)
                    }
                    (false, false) => {
                        self.reporter.show_error(
                            "illegal pattern",
                            &[
                                (
                                    p1.span(),
                                    "expected literal, cons or tuple constructor here",
                                ),
                                (
                                    p2.span(),
                                    "expected literal, cons or tuple constructor here",
                                ),
                            ],
                        );
                        Err(NoMatchError)
                    }
                    _ => {
                        match (p1, p2) {
                            (Expr::Literal(l1), Expr::Literal(l2)) if l1 == l2 => {
                                Ok(Expr::Literal(l1))
                            }
                            (
                                Expr::Literal(Literal::String(s)),
                                nil @ Expr::Literal(Literal::Nil(_)),
                            )
                            | (
                                nil @ Expr::Literal(Literal::Nil(_)),
                                Expr::Literal(Literal::String(s)),
                            ) if s.name == symbols::Empty => Ok(nil),
                            (
                                p1 @ Expr::Literal(Literal::Nil(_)),
                                Expr::Literal(Literal::Nil(_)),
                            ) => Ok(p1),
                            (p1 @ Expr::Cons(_), p2 @ Expr::Cons(_)) => {
                                // TODO: Implement data_elements
                                let elements1 = p1.data_elements();
                                let elements2 = p2.data_elements();
                                if elements1.len() != elements2.len() {
                                    self.reporter.show_error(
                                        "impossible match",
                                        &[
                                            (p1.span(), "this pattern can never match"),
                                            (p2.span(), "as it is incompatible with this pattern"),
                                        ],
                                    );
                                    return Err(NoMatchError);
                                }
                                let elements = elements1
                                    .drain(..)
                                    .zip(elements2.drain(..))
                                    .map(|(a, b)| self.normalize_alias_patterns(a, b));
                                Ok(Expr::make_list(elements))
                            }
                            (p1 @ Expr::Tuple(_), p2 @ Expr::Tuple(_)) => {
                                let elements1 = p1.data_elements();
                                let elements2 = p2.data_elements();
                                if elements1.len() != elements2.len() {
                                    self.reporter.show_error(
                                        "impossible match",
                                        &[
                                            (p1.span(), "this pattern can never match"),
                                            (p2.span(), "as it is incompatible with this pattern"),
                                        ],
                                    );
                                    return Err(NoMatchError);
                                }
                                let elements = elements1
                                    .drain(..)
                                    .zip(elements2.drain(..))
                                    .map(|(a, b)| self.normalize_alias_patterns(a, b))
                                    .collect();
                                Ok(Expr::Tuple(Tuple {
                                    span: p1.span(),
                                    elements,
                                }))
                            }
                            (p1, p2) => {
                                self.reporter.show_error(
                                    "impossible match",
                                    &[
                                        (p1.span(), "this pattern can never match"),
                                        (p2.span(), "as it is incompatible with this pattern"),
                                    ],
                                );
                                Err(NoMatchError)
                            }
                        }
                    }
                }
            }
        }
    }

    fn normalize_alias_map_pairs<I: Iterator<Item = MapField>>(
        &mut self,
        fields: I,
    ) -> Result<Vec<MapField>, NoMatchError> {
        use std::collections::btree_map::Entry;

        let mut aliased_pairs = BTreeMap::new::<MapSortKey, VecDeque<MapField>>();
        for field in fields {
            let key = map_sort_key(field.key(), &aliased_pairs);
            match aliased_pairs.entry(key) {
                Entry::Occupied(entry) => {
                    entry.get_mut().push_back(field);
                }
                Entry::Vacant(entry) => {
                    entry.insert(VecDeque::from([field]));
                }
            }
        }

        let mut normalized = Vec::with_capacity(aliased_pairs.len());
        for mut fields in aliased_pairs.into_values() {
            let first = fields.pop_front().unwrap();
            let span = first.span;
            let key = first.key;
            let mut pattern = Some(first.value);
            for field in fields.drain(..) {
                pattern =
                    Some(self.normalize_alias_patterns(field.value, pattern.take().unwrap())?);
            }
            normalized.push(MapField::Exact {
                span,
                key,
                value: pattern.take().unwrap(),
            });
        }

        Ok(normalized)
    }
}

/// This struct is used to represent the internal state of a comprehension qualifier while it is being transformed
enum Qualifier {
    Gen(SourceSpan, Gen),
    Filter(SourceSpan, Filter)
}

impl CanonicalizeFuntion {
    //  Preprocess a list of Erlang qualifiers into its intermediate representation,
    //  represented as a list of #igen{} and #ifilter{} records. We recognise guard
    //  tests and try to fold them together and join to a preceding generators, this
    //  should give us better and more compact code.
    fn preprocess_quals(&mut self, mut qualifiers: Vec<Expr>) -> Vec<Qualifiers> {
        let mut output = Vec::with_capacity(qualifiers.len());

        let mut it = qualifiers.drain(..);

        while let Some(q) = it.next() {
            let span = q.span();
            match q {
                Expr::Generator(gen) => {
                    let mut guards = vec![];
                    while let Some(q2) = it.next() {
                        if q2.is_guard_test() {
                            guards.push(q2);
                            continue;
                        }
                        break;
                    }
                    output.push(Qualifier::Gen(span, self.generator(gen, guards)?));
                }
                q if q.is_guard_test() => {
                    let mut guards = vec![q];
                    while let Some(q2) = it.next() {
                        if q2.is_guard_test() {
                            guards.push(q2);
                            continue;
                        }
                        break;
                    }
                    let guards = self.lc_guard_tests(guards);
                    output.push(Qualifier::Filter(span, Filter::Guard(guards)));
                }
                q => {
                    let (expr, pre) = self.novars(q);
                    output.push(Qualifier::Filter(span, Filter::Pattern(pre, expr)));
                }
            }
        }

        output
    }
}

/// This struct is used to represent the internal state of a generator expression while it is being transformed
struct Gen {
    // acc_pat is the accumulator pattern, e.g. [Pat|Tail] for Pat <- Expr.
    acc_pattern: Expr,
    // acc_guard is the list of guards immediately following the current
    // generator in the qualifier list input.
    acc_guard: Vec<Expr>,
    // skip_pat is the skip pattern, e.g. <<X,_:X,Tail/bitstring>> for <<X,1:X>> <= Expr.
    skip_pat: Expr,
    // tail is the variable used in AccPat and SkipPat bound to the rest of the
    // generator input.
    tail: Var,
    // tail_pat is the tail pattern, respectively [] and <<_/bitstring>> for list
    // and bit string generators.
    tail_pat: Expr,
    // pre is the list of expressions to be inserted before the comprehension function
    pre: Vec<Expr>,
    // arg is the expression that the comprehension function should be passed
    arg: Expr,
}

impl CanonicalizeFunction {
    fn generator(&mut self, generator: Generator, guards: Vec<Expr>) -> Gen {
        todo!()
    }
}

/// A filter is one of two types of expressions that act as qualifiers in a commprehension, the other is a generator
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    Guard(Vec<Expr>),
    /// Represents a filter expression which lowers to a case
    ///
    /// The first element is used to guarantee that certain expressions
    /// are evaluated before the filter is applied
    ///
    /// The second element is the argument on which the filter is matched
    /// It must be true to accumulate the current value
    /// If false, the current value is skipped
    /// If neither, a bad_filter error is raised
    Match(Vec<Expr>, Box<Expr>),
}

impl CanonicalizeFunction {
    fn lc_guard_tests(&mut self, guards: Vec<Expr>) -> Vec<Expr> {
        let tests = self.guard_tests(guards);
        self.in_guard = true;
        let guards = self.gexpr_top(tests);
        self.in_guard = false;
        guards
    }

    fn guard_tests(&mut self, mut guards: Vec<Expr>) -> Expr {
        let last = guards.pop();
        let guard = guards.drain(..).rfold(last, |guard, acc| {
            let span = guard.span();
            Expr::BinaryExpr(BinaryExpr { span, op: BinaryOp::And, lhs: Box::new(guard), rhs: Box::new(acc) })
        });
        Expr::Protect(guard)
    }

    /// Generate an internal core expression of a guard tests.
    /// Explicitly handle outer boolean expressions and "protect" inner tests
    fn gexpr_top(&mut self, expr: Expr) -> Vec<Expr> {
        let (expr, pre, bools) = self.gexpr(expr, vec![]);
        let (expr, mut pre) = self.force_booleans(bools, expr, pre);
        pre.push(expr);
        pre
    }

    fn gexpr(&mut self, expr: Expr, bools: Vec<Expr>) -> (Expr, Vec<Expr>, Vec<Expr>) {
        match expr {
            Expr::Protect(expr) => {
                let (expr, pre, bools2) = self.gexpr(*expr, vec![]);
                if pre.is_empty() => {
                    let (expr, pre) = self.force_booleans(bools2, expr, vec![]);
                    (expr, pre, bools)
                } else {
                    let (expr, mut pre) = self.force_booleans(bools2, expr, pre);
                    pre.push(expr);
                    (Expr::ProtectAll(pre), vec![], bools)
                }
            }
        }
    }
}

fn map_sort_key(key: Expr, keys: &BTreeMap<MapSortKey, Vec<Expr>>) -> MapSortKey {
    match key {
        Expr::Literal(lit) => MapSortKey::Atomic(lit),
        Expr::Var(var) => MapSortKey::Var(var),
        _ => MapSortKey::Expr(keys.len()),
    }
}

#[derive(Clone)]
enum MapSortKey {
    Atomic(Literal),
    Var(Var),
    Expr(usize),
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
            (Self::Atomic(a), Self::Atomic(b)) => a.cmp(b),
            (Self::Atomic(a), _) => Ordering::Less,
            (Self::Var(a), Self::Var(b)) => a.cmp(b),
            (Self::Var(_), Self::Atomic(_)) => Ordering::Greater,
            (Self::Var(_), _) => Ordering::Less,
            (Self::Expr(a), Self::Expr(b)) => a.cmp(b),
            (Self::Expr(_), _) => Ordering::Greater,
        }
    }
}

