///! Kernel erlang is like Core Erlang with a few significant
///! differences:
///!
///! 1. It is flat!  There are no nested calls or sub-blocks.
///!
///! 2. All variables are unique in a function.  There is no scoping, or
///! rather the scope is the whole function.
///!
///! 3. Pattern matching (in cases and receives) has been compiled.
///!
///! 4. All remote-calls are to statically named m:f/a. Meta-calls are
///! passed via erlang:apply/3.
///!
///! The translation is done in two passes:
///!
///! 1. Basic translation, translate variable/function names, flatten
///! completely, pattern matching compilation.
///!
///! 2. Fun-lifting (lambda-lifting), variable usage annotation and
///! last-call handling.
///!
///! All new Kexprs are created in the first pass, they are just
///! annotated in the second.
///!
///! Functions and BIFs
///!
///! Functions are "call"ed or "enter"ed if it is a last call, their
///! return values may be ignored.  BIFs are things which are known to
///! be internal by the compiler and can only be called, their return
///! values cannot be ignored.
///!
///! Letrec's are handled rather naively.  All the functions in one
///! letrec are handled as one block to find the free variables.  While
///! this is not optimal it reflects how letrec's often are used.  We
///! don't have to worry about variable shadowing and nested letrec's as
///! this is handled in the variable/function name translation.  There
///! is a little bit of trickery to ensure letrec transformations fit
///! into the scheme of things.
///!
///! To ensure unique variable names we use a variable substitution
///! table and keep the set of all defined variables.  The nested
///! scoping of Core means that we must also nest the substitution
///! tables, but the defined set must be passed through to match the
///! flat structure of Kernel and to make sure variables with the same
///! name from different scopes get different substitutions.
///!
///! We also use these substitutions to handle the variable renaming
///! necessary in pattern matching compilation.
///!
///! The pattern matching compilation assumes that the values of
///! different types don't overlap.  This means that as there is no
///! character type yet in the machine all characters must be converted
///! to integers!
use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::ops::RangeInclusive;
use std::sync::Arc;

use rpds::{rbt_set, RedBlackTreeSet};

use firefly_binary::{BinaryEntrySpecifier, BitVec, Endianness};
use firefly_intern::{symbols, Ident, Symbol};
use firefly_number::Int;
use firefly_pass::Pass;
use firefly_syntax_base::*;
use firefly_syntax_core as core;
use firefly_util::diagnostics::*;

use crate::{passes::FunctionContext, *};

// Matches collapse max segment CST translation.
const EXPAND_MAX_SIZE_SEGMENT: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum ExprError {
    #[error("bad segment size")]
    BadSegmentSize(SourceSpan),
}

/// This pass transforms a Core IR function into its Kernel IR form for further analysis and
/// eventual lowering to SSA IR
pub struct CoreToKernel {
    diagnostics: Arc<DiagnosticsHandler>,
}
impl CoreToKernel {
    pub fn new(diagnostics: Arc<DiagnosticsHandler>) -> Self {
        Self { diagnostics }
    }
}
impl Pass for CoreToKernel {
    type Input<'a> = core::Module;
    type Output<'a> = Module;

    fn run<'a>(&mut self, mut cst: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let mut module = Module {
            span: cst.span,
            annotations: Annotations::default(),
            name: cst.name,
            compile: cst.compile,
            on_load: cst.on_load,
            nifs: cst.nifs,
            exports: cst.exports,
            functions: vec![],
        };

        let mut funs = vec![];

        while let Some((name, function)) = cst.functions.pop_first() {
            let context = FunctionContext::new(function.span(), name, function.var_counter);

            let mut pipeline = TranslateCore::new(&self.diagnostics, context, module.name.name);
            let fun = pipeline.run(function.fun)?;
            module.functions.push(fun);
            funs.append(&mut pipeline.context.funs);
        }

        module.functions.append(&mut funs);

        Ok(module)
    }
}

struct TranslateCore<'p> {
    diagnostics: &'p DiagnosticsHandler,
    context: FunctionContext,
    module_name: Symbol,
}
impl<'p> TranslateCore<'p> {
    fn new(
        diagnostics: &'p DiagnosticsHandler,
        context: FunctionContext,
        module_name: Symbol,
    ) -> Self {
        Self {
            diagnostics,
            context,
            module_name,
        }
    }
}
impl<'p> Pass for TranslateCore<'p> {
    type Input<'a> = core::Fun;
    type Output<'a> = Function;

    fn run<'a>(&mut self, fun: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        match self.expr(core::Expr::Fun(fun), BiMap::default())? {
            (Expr::Fun(ifun), _) => {
                let span = ifun.span();
                let annotations = ifun.annotations;
                let vars = ifun.vars;
                let (body, _) = self.ubody(*ifun.body, Brk::Return)?;
                match body {
                    body @ Expr::Match(_) => Ok(Function {
                        span,
                        annotations,
                        name: self.context.name,
                        vars,
                        body: Box::new(body),
                    }),
                    other => {
                        let body = Box::new(Expr::Match(Match {
                            span,
                            annotations: Annotations::default(),
                            body: Box::new(other),
                            ret: vec![],
                        }));
                        Ok(Function {
                            span,
                            annotations,
                            name: self.context.name,
                            vars,
                            body,
                        })
                    }
                }
            }
            _ => unreachable!(),
        }
    }
}
impl<'p> TranslateCore<'p> {
    /// body(Cexpr, Sub, State) -> {Kexpr,[PreKepxr],State}.
    ///  Do the main sequence of a body.  A body ends in an atomic value or
    ///  values.  Must check if vector first so do expr.
    fn body(&mut self, expr: core::Expr, sub: BiMap) -> Result<(Expr, Vec<Expr>), ExprError> {
        match expr {
            core::Expr::Values(core::Values {
                annotations,
                values,
                ..
            }) => {
                // Do this here even if only in bodies
                let (values, pre) = self.atomic_list(values, sub)?;
                Ok((
                    Expr::Values(IValues {
                        annotations,
                        values,
                    }),
                    pre,
                ))
            }
            expr => self.expr(expr, sub),
        }
    }

    /// guard(Cexpr, Sub, State) -> {Kexpr,State}.
    ///  We handle guards almost as bodies. The only special thing we
    ///  must do is to make the final Kexpr a #k_test{}.
    fn guard(&mut self, expr: core::Expr, sub: BiMap) -> Result<Expr, ExprError> {
        let (guard, pre) = self.expr(expr, sub)?;
        let guard = self.gexpr_test(guard);
        Ok(pre_seq(pre, guard))
    }

    /// gexpr_test(Kexpr, State) -> {Kexpr,State}.
    ///  Builds the final boolean test from the last Kexpr in a guard test.
    ///  Must enter try blocks and isets and find the last Kexpr in them.
    ///  This must end in a recognised BEAM test!
    fn gexpr_test(&mut self, expr: Expr) -> Expr {
        match expr {
            // Convert to test
            Expr::Bif(bif) if bif.is_type_test() || bif.is_comp_op() => {
                let span = bif.span;
                let annotations = bif.annotations;
                let op = bif.op;
                let args = bif.args;
                Expr::Test(Test {
                    span,
                    annotations,
                    op,
                    args,
                })
            }
            Expr::Try(Try {
                span,
                annotations,
                arg,
                vars,
                body,
                evars,
                handler,
                ret,
            }) => {
                let arg = self.gexpr_test(*arg);
                Expr::Try(Try {
                    span,
                    annotations,
                    arg: Box::new(arg),
                    vars,
                    body,
                    evars,
                    handler,
                    ret,
                })
            }
            Expr::Set(ISet {
                span,
                annotations,
                vars,
                arg,
                body,
            }) => {
                let body = self.gexpr_test(*body.unwrap());
                Expr::Set(ISet {
                    span,
                    annotations,
                    vars,
                    arg,
                    body: Some(Box::new(body)),
                })
            }
            // Add equality test
            expr => self.gexpr_test_add(expr),
        }
    }

    fn gexpr_test_add(&mut self, expr: Expr) -> Expr {
        let span = expr.span();
        let annotations = expr.annotations().clone();
        let op = FunctionName::new(symbols::Erlang, symbols::EqualStrict, 2);
        let (expr, pre) = self.force_atomic(expr);
        let t = Expr::Literal(Literal::atom(span, symbols::True));
        pre_seq(
            pre,
            Expr::Test(Test {
                span,
                annotations,
                op,
                args: vec![expr, t],
            }),
        )
    }

    /// Convert a core expression to a kernel expression, flattening it.
    fn expr(&mut self, expr: core::Expr, sub: BiMap) -> Result<(Expr, Vec<Expr>), ExprError> {
        match expr {
            core::Expr::Var(v) if v.arity.is_some() => {
                let span = v.span();
                let arity = v.arity.unwrap();
                let name = Name::from(&v);
                let name = sub.get(name).unwrap_or(name);
                let local = FunctionName::new_local(name.symbol(), arity as u8);
                Ok((Expr::Local(Span::new(span, local)), vec![]))
            }
            core::Expr::Var(mut v) => {
                let name = sub.get_vsub(v.name.name);
                v.name.name = name;
                Ok((Expr::Var(v), vec![]))
            }
            core::Expr::Literal(lit) => Ok((Expr::Literal(lit), vec![])),
            core::Expr::Cons(core::Cons {
                span,
                annotations,
                box head,
                box tail,
            }) => {
                // Do cons in two steps, first the expressions left to right,
                // then any remaining literals right to left.
                let (kh, mut pre) = self.expr(head, sub.clone())?;
                let (kt, mut pre2) = self.expr(tail, sub.clone())?;
                let (kt, mut pre3) = self.force_atomic(kt);
                let (kh, mut pre4) = self.force_atomic(kh);
                pre.append(&mut pre2);
                pre.append(&mut pre3);
                pre.append(&mut pre4);
                Ok((
                    Expr::Cons(Cons {
                        span,
                        annotations,
                        head: Box::new(kh),
                        tail: Box::new(kt),
                    }),
                    pre,
                ))
            }
            core::Expr::Tuple(core::Tuple {
                span,
                annotations,
                elements,
            }) => {
                let (elements, pre) = self.atomic_list(elements, sub)?;
                Ok((
                    Expr::Tuple(Tuple {
                        span,
                        annotations,
                        elements,
                    }),
                    pre,
                ))
            }
            core::Expr::Map(core::Map {
                annotations,
                box arg,
                pairs,
                ..
            }) => self.expr_map(annotations, arg, pairs, sub),
            core::Expr::Binary(core::Binary {
                span,
                annotations,
                segments,
            }) => match self.atomic_bin(segments, sub.clone()) {
                Ok((segment, pre)) => Ok((
                    Expr::Binary(Binary {
                        span,
                        annotations,
                        segment,
                    }),
                    pre,
                )),
                Err(ExprError::BadSegmentSize(span)) => {
                    self.diagnostics
                        .diagnostic(Severity::Error)
                        .with_message("invalid binary size")
                        .with_primary_label(span, "associated with this segment")
                        .emit();
                    let badarg = core::Expr::Literal(Literal::atom(span, symbols::Badarg));
                    let error =
                        core::Call::new(span, symbols::Erlang, symbols::Error, vec![badarg]);
                    self.expr(core::Expr::Call(error), sub)
                }
            },
            core::Expr::Fun(core::Fun {
                span,
                annotations,
                mut vars,
                box body,
                ..
            }) => {
                // Build up the set of current fun arguments
                let cvars = vars.drain(..).map(core::Expr::Var).collect();
                let (mut vars, sub) = self.pattern_list(cvars, sub.clone(), sub)?;
                let vars = vars.drain(..).map(|v| v.into_var().unwrap()).collect();
                // Save any parent fun arguments, replacing with the current args
                let parent_vars = mem::replace(&mut self.context.args, vars);
                let (body, pre) = self.body(body, sub)?;
                let body = pre_seq(pre, body);
                // Place back the original fun arguments
                let vars = mem::replace(&mut self.context.args, parent_vars);
                Ok((
                    Expr::Fun(IFun {
                        span,
                        annotations,
                        vars,
                        body: Box::new(body),
                    }),
                    vec![],
                ))
            }
            core::Expr::Seq(core::Seq {
                box arg, box body, ..
            }) => {
                let (arg, mut pre) = self.body(arg, sub.clone())?;
                let (body, mut pre2) = self.body(body, sub)?;
                pre.push(arg);
                pre.append(&mut pre2);
                Ok((body, pre))
            }
            core::Expr::Let(core::Let {
                span,
                annotations,
                mut vars,
                box arg,
                box body,
            }) => {
                let (arg, mut pre) = self.body(arg, sub.clone())?;
                let cvars = vars.drain(..).map(core::Expr::Var).collect();
                let (mut vars, sub) = self.pattern_list(cvars, sub.clone(), sub)?;
                // Break down multiple values into separate set expressions
                match arg {
                    Expr::Values(IValues { mut values, .. }) => {
                        assert_eq!(vars.len(), values.len());
                        pre.extend(vars.drain(..).zip(values.drain(..)).map(|(var, value)| {
                            let var = var.into_var().unwrap();
                            let span = var.span();
                            kset!(span, var, value).into()
                        }));
                    }
                    arg => {
                        pre.push(Expr::Set(ISet {
                            span,
                            annotations,
                            vars: vars.drain(..).map(|v| v.into_var().unwrap()).collect(),
                            arg: Box::new(arg),
                            body: None,
                        }));
                    }
                };
                let (body, mut pre2) = self.body(body, sub)?;
                pre.append(&mut pre2);
                Ok((body, pre))
            }
            core::Expr::LetRec(core::LetRec {
                span,
                annotations,
                mut defs,
                box body,
            }) => {
                if annotations.contains(symbols::LetrecGoto) {
                    assert_eq!(defs.len(), 1);
                    let (var, def) = defs.pop().unwrap();
                    self.letrec_goto(span, var, def, body, sub)
                } else {
                    self.letrec_local_function(span, annotations, defs, body, sub)
                }
            }
            core::Expr::Case(core::Case {
                box arg, clauses, ..
            }) => {
                let (arg, mut pre) = self.body(arg, sub.clone())?;
                let (vars, mut pre2) = self.match_vars(arg);
                let mexpr = self.kmatch(vars, clauses, sub)?;
                let mut seq = flatten_seq(build_match(mexpr));
                let expr = seq.pop().unwrap();
                pre.append(&mut pre2);
                pre.append(&mut seq);
                Ok((expr, pre))
            }
            core::Expr::If(core::If {
                span,
                annotations,
                box guard,
                box then_body,
                box else_body,
            }) => {
                // Convert the original expression into the following form:
                //
                //   let <cond> = <guard>
                //    in if <cond> then <then_body>
                //                 else <else_body>
                //
                // This ensures that the conditional of the `if` is always
                // a variable containing the result of an expression or a literal
                let cond_span = guard.span();
                let (cond, pre) = self.body(guard, sub.clone())?;
                let cond = pre_seq(pre, cond);

                let (then_body, tpre) = self.body(then_body, sub.clone())?;
                let then_body = pre_seq(tpre, then_body);
                let (else_body, epre) = self.body(else_body, sub.clone())?;
                let else_body = pre_seq(epre, else_body);

                let cond_var = self.context.next_var(Some(cond_span));
                let body = Expr::If(If {
                    span,
                    annotations,
                    cond: Box::new(Expr::Var(cond_var.clone())),
                    then_body: Box::new(then_body),
                    else_body: Box::new(else_body),
                    ret: vec![],
                });
                let set = Expr::Set(ISet::new(
                    cond_span,
                    vec![cond_var.clone()],
                    cond,
                    Some(body),
                ));

                Ok((set, vec![]))
            }
            core::Expr::Apply(core::Apply {
                span,
                annotations,
                box callee,
                args,
            }) => self.capply(span, annotations, callee, args, sub),
            core::Expr::Call(core::Call {
                span,
                annotations,
                box module,
                box function,
                mut args,
            }) => {
                match call_type(&module, &function, args.as_slice()) {
                    CallType::Error => {
                        // Invalid module/function, must rewrite as a call to apply/3 and let it
                        // fail at runtime
                        let argv = make_clist(args);
                        let mut mfa = Vec::with_capacity(3);
                        mfa.push(module);
                        mfa.push(function);
                        mfa.push(argv);
                        let call = core::Expr::Call(core::Call::new(
                            span,
                            symbols::Erlang,
                            symbols::Apply,
                            mfa,
                        ));
                        self.expr(call, sub)
                    }
                    CallType::Bif(op) => {
                        let (args, pre) = self.atomic_list(args, sub)?;
                        Ok((
                            Expr::Bif(Bif {
                                span,
                                annotations,
                                op,
                                args,
                                ret: vec![],
                            }),
                            pre,
                        ))
                    }
                    CallType::Static(callee) => {
                        let (args, pre) = self.atomic_list(args, sub)?;
                        Ok((
                            Expr::Call(Call {
                                span,
                                annotations,
                                callee: Box::new(Expr::Remote(Remote::Static(Span::new(
                                    span, callee,
                                )))),
                                args,
                                ret: vec![],
                            }),
                            pre,
                        ))
                    }
                    CallType::Dynamic => {
                        let mut mfa = Vec::with_capacity(args.len() + 2);
                        mfa.push(module);
                        mfa.push(function);
                        mfa.append(&mut args);
                        let (mut mfa, pre) = self.atomic_list(mfa, sub)?;
                        let args = mfa.split_off(2);
                        let function = mfa.pop().unwrap();
                        let module = mfa.pop().unwrap();
                        let callee = Box::new(Expr::Remote(Remote::Dynamic(
                            Box::new(module),
                            Box::new(function),
                        )));
                        Ok((
                            Expr::Call(Call {
                                span,
                                annotations,
                                callee,
                                args,
                                ret: vec![],
                            }),
                            pre,
                        ))
                    }
                }
            }
            core::Expr::PrimOp(core::PrimOp {
                span,
                annotations,
                name: symbols::MatchFail,
                mut args,
            }) => self.translate_match_fail(span, annotations, args.pop().unwrap(), sub),
            core::Expr::PrimOp(core::PrimOp {
                span,
                name: symbols::MakeFun,
                args,
                ..
            }) if args.len() == 3 && args.iter().all(|arg| arg.is_literal()) => {
                // If make_fun/3 is called with all literal values, convert it to its ideal form
                match args.as_slice() {
                    [core::Expr::Literal(Literal {
                        value: Lit::Atom(m),
                        ..
                    }), core::Expr::Literal(Literal {
                        value: Lit::Atom(f),
                        ..
                    }), core::Expr::Literal(Literal {
                        value: Lit::Integer(Int::Small(arity)),
                        ..
                    })] => {
                        let arity = *arity;
                        if *m == self.module_name {
                            let local = Expr::Local(Span::new(
                                span,
                                FunctionName::new_local(*f, arity.try_into().unwrap()),
                            ));
                            let op = FunctionName::new(symbols::Erlang, symbols::MakeFun, 3);
                            Ok((Expr::Bif(Bif::new(span, op, vec![local])), vec![]))
                        } else {
                            let remote = Expr::Remote(Remote::Static(Span::new(
                                span,
                                FunctionName::new(*m, *f, arity.try_into().unwrap()),
                            )));
                            let op = FunctionName::new(symbols::Erlang, symbols::MakeFun, 3);
                            Ok((Expr::Bif(Bif::new(span, op, vec![remote])), vec![]))
                        }
                    }
                    other => panic!("invalid callee, expected literal mfa, got {:#?}", &other),
                }
            }
            core::Expr::PrimOp(core::PrimOp {
                span, name, args, ..
            }) => {
                let arity = args.len() as u8;
                let (args, pre) = self.atomic_list(args, sub)?;
                let op = FunctionName::new(symbols::Erlang, name, arity);
                Ok((Expr::Bif(Bif::new(span, op, args)), pre))
            }
            core::Expr::Try(core::Try {
                span,
                annotations,
                box arg,
                mut vars,
                box body,
                mut evars,
                box handler,
            }) => {
                // The normal try expression. The body and exception handler
                // variables behave as let variables.
                let (arg, apre) = self.body(arg, sub.clone())?;
                let cvars = vars.drain(..).map(core::Expr::Var).collect();
                let (mut vars, sub1) = self.pattern_list(cvars, sub.clone(), sub.clone())?;
                let vars = vars.drain(..).map(|v| v.into_var().unwrap()).collect();
                let (body, bpre) = self.body(body, sub1)?;
                let cevars = evars.drain(..).map(core::Expr::Var).collect();
                let (mut evars, sub2) = self.pattern_list(cevars, sub.clone(), sub)?;
                let evars = evars.drain(..).map(|v| v.into_var().unwrap()).collect();
                let (handler, hpre) = self.body(handler, sub2)?;
                let arg = pre_seq(apre, arg);
                let body = pre_seq(bpre, body);
                let handler = pre_seq(hpre, handler);
                Ok((
                    Expr::Try(Try {
                        span,
                        annotations,
                        arg: Box::new(arg),
                        body: Box::new(body),
                        handler: Box::new(handler),
                        vars,
                        evars,
                        ret: vec![],
                    }),
                    vec![],
                ))
            }
            core::Expr::Catch(core::Catch {
                span,
                annotations,
                box body,
            }) => {
                let (body, pre) = self.body(body, sub)?;
                let body = pre_seq(pre, body);
                Ok((
                    Expr::Catch(Catch {
                        span,
                        annotations,
                        body: Box::new(body),
                        ret: vec![],
                    }),
                    vec![],
                ))
            }

            other => panic!("untranslatable core expression for kernel: {:?}", &other),
        }
    }

    /// Implement letrec in the traditional way as a local function for each definition in the
    /// letrec
    fn letrec_local_function(
        &mut self,
        span: SourceSpan,
        annotations: Annotations,
        mut defs: Vec<(Var, core::Expr)>,
        body: core::Expr,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        // Make new function names and store substitution
        let sub = defs
            .iter_mut()
            .fold(sub, |sub, (ref mut var, ref mut def)| {
                let arity = var.arity.take().unwrap();
                let span = var.name.span;
                let f = var.name.name;
                let ty = format!("{}/{}", f, arity);
                let n = self.context.new_fun_name(Some(&ty));
                def.annotate(symbols::LetrecName, Literal::atom(span, n));
                var.name = Ident::new(n, span);
                sub.set_fsub(f, arity, Name::Var(n))
            });
        // Run translation on functions and body.
        let defs = defs
            .drain(..)
            .map(|(var, def)| {
                let (Expr::Fun(mut def), dpre) = self.expr(def, sub.clone())? else { panic!("expected ifun") };
                assert_eq!(dpre.len(), 0);
                def.annotations_mut().replace(annotations.clone());
                Ok((var, def))
            })
            .try_collect()?;
        let (body, mut bpre) = self.body(body, sub)?;
        let mut pre = vec![Expr::LetRec(ILetRec {
            span,
            annotations,
            defs,
        })];
        pre.append(&mut bpre);
        Ok((body, pre))
    }

    /// Implement letrec with the sole definition as a label and each apply of it as a goto
    fn letrec_goto(
        &mut self,
        span: SourceSpan,
        var: Var,
        fail: core::Expr,
        body: core::Expr,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        let label = var.name.name;
        assert_ne!(var.arity, None);
        let core::Expr::Fun(fail) = fail else { panic!("unexpected letrec definition") };
        let fun_vars = fail.vars;
        let fun_body = fail.body;
        let mut kvars = Vec::with_capacity(fun_vars.len());
        let fun_sub = fun_vars.iter().fold(sub.clone(), |sub, fv| {
            let new_name = self.context.next_var_name(Some(fv.span()));
            kvars.push(Var {
                annotations: fv.annotations.clone(),
                name: new_name,
                arity: None,
            });
            let sub = sub.set_vsub(fv.name.name, new_name.name);
            self.context.defined.insert_mut(new_name);
            sub
        });
        let labels0 = self.context.labels.clone();
        self.context.labels.insert_mut(label);
        let (body, bpre) = self.body(body, sub.clone())?;
        let (fail, fpre) = self.body(*fun_body, fun_sub)?;
        match (body, fail) {
            (Expr::Goto(gt), Expr::Goto(inner_gt)) if gt.label == label => {
                Ok((Expr::Goto(inner_gt), bpre))
            }
            (body, fail) => {
                self.context.labels = labels0;
                let then_body = pre_seq(fpre, fail);
                let alt = Expr::LetRecGoto(LetRecGoto {
                    span,
                    annotations: Annotations::default(),
                    label,
                    vars: kvars,
                    first: Box::new(body),
                    then: Box::new(then_body),
                    ret: vec![],
                });
                Ok((alt, bpre))
            }
        }
    }

    /// Translate match_fail primop, paying extra attention to `function_clause`
    /// errors that may have been inlined from other functions.
    fn translate_match_fail(
        &mut self,
        span: SourceSpan,
        annotations: Annotations,
        arg: core::Expr,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        let (args, annotations) = match arg {
            core::Expr::Tuple(core::Tuple {
                annotations,
                elements,
                ..
            }) => match &elements[0] {
                core::Expr::Literal(Literal {
                    value: Lit::Atom(symbols::FunctionClause),
                    ..
                }) => self.translate_fc_args(elements, sub.clone(), annotations),
                _ => (elements, annotations),
            },
            core::Expr::Literal(Literal {
                value: Lit::Tuple(mut elements),
                ..
            }) => match &elements[0].value {
                Lit::Atom(symbols::FunctionClause) => {
                    let args = elements.drain(..).map(core::Expr::Literal).collect();
                    self.translate_fc_args(args, sub.clone(), annotations)
                }
                _ => {
                    let args = elements.drain(..).map(core::Expr::Literal).collect();
                    (args, annotations)
                }
            },
            reason @ core::Expr::Literal(_) => (vec![reason], annotations),
            other => panic!("unexpected match_fail argument type: {:?}", &other),
        };
        let arity = args.len();
        let (args, pre) = self.atomic_list(args, sub)?;
        Ok((
            Expr::Bif(Bif {
                span,
                annotations,
                op: FunctionName::new(symbols::Erlang, symbols::MatchFail, arity as u8),
                args,
                ret: vec![],
            }),
            pre,
        ))
    }

    fn translate_fc_args(
        &mut self,
        args: Vec<core::Expr>,
        sub: BiMap,
        mut annotations: Annotations,
    ) -> (Vec<core::Expr>, Annotations) {
        if same_args(args.as_slice(), self.context.args.as_slice(), &sub) {
            // The arguments for the function_clause exception are the arguments
            // for the current function in the correct order.
            (args, annotations)
        } else {
            // The arguments in the function_clause exception don't match the
            // arguments for the current function because of inlining
            match annotations.get(symbols::Function) {
                None => {
                    let span = SourceSpan::default();
                    let name = self.context.new_fun_name(Some("inlined"));
                    let name = Literal::atom(span, name);
                    let arity = Literal::integer(span, args.len() - 1);
                    let na = Literal::tuple(span, vec![name, arity]);
                    annotations.insert_mut(symbols::Inlined, na);
                    (args, annotations)
                }
                Some(Annotation::Term(Literal {
                    value: Lit::Tuple(elems),
                    ..
                })) if elems.len() == 2 => {
                    // This is the function that was inlined
                    let span = SourceSpan::default();
                    let name0 = elems[0].as_atom().unwrap();
                    let arity0 = elems[1].as_integer().unwrap();
                    let name = format!("-inlined-{}/{}-", name0, arity0);
                    let name = Literal::atom(span, Symbol::intern(&name));
                    let arity = Literal::integer(span, arity0.clone());
                    let na = Literal::tuple(span, vec![name, arity]);
                    annotations.insert_mut(symbols::Inlined, na);
                    (args, annotations)
                }
                other => panic!("unexpected value for 'function' annotation: {:?}", &other),
            }
        }
    }

    fn expr_map(
        &mut self,
        annotations: Annotations,
        arg: core::Expr,
        pairs: Vec<core::MapPair>,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        let (var, mut pre) = self.expr(arg, sub.clone())?;
        let (m, mut pre2) = self.map_split_pairs(annotations, var, pairs, sub)?;
        pre2.append(&mut pre);
        Ok((m, pre2))
    }

    fn map_split_pairs(
        &mut self,
        annotations: Annotations,
        var: Expr,
        mut pairs: Vec<core::MapPair>,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        // 1. Force variables.
        // 2. Group adjacent pairs with literal keys.
        // 3. Within each such group, remove multiple assignments to the same key.
        // 4. Partition each group according to operator ('=>' and ':=')
        let mut pre = vec![];
        let mut kpairs = Vec::with_capacity(pairs.len());

        for core::MapPair {
            op,
            box key,
            box value,
        } in pairs.drain(..)
        {
            let (key, mut pre1) = self.atomic(key, sub.clone())?;
            let (value, mut pre2) = self.atomic(value, sub.clone())?;
            kpairs.push((
                op,
                MapPair {
                    key: Box::new(key),
                    value: Box::new(value),
                },
            ));
            pre.append(&mut pre1);
            pre.append(&mut pre2);
        }

        let mut iter = kpairs.drain(..);
        let mut groups: Vec<(MapOp, Vec<MapPair>)> = vec![];

        // Group adjacent pairs with literal keys, variables are always in their own groups
        let mut seen: BTreeMap<Literal, (MapOp, MapPair)> = BTreeMap::new();
        while let Some((op, pair)) = iter.next() {
            match pair.key.as_ref() {
                Expr::Var(_) => {
                    // Do not group variable keys with other keys
                    if !seen.is_empty() {
                        // Partition the group by operator
                        let (mut a, mut b) = seen
                            .into_values()
                            .partition::<Vec<_>, _>(|(op, _pair)| op == &MapOp::Exact);
                        if !a.is_empty() {
                            groups.push((MapOp::Exact, a.drain(..).map(|(_, p)| p).collect()));
                        }
                        if !b.is_empty() {
                            groups.push((MapOp::Assoc, b.drain(..).map(|(_, p)| p).collect()));
                        }
                        groups.push((op, vec![pair]));
                        seen = BTreeMap::new();
                    } else {
                        groups.push((op, vec![pair]));
                    }
                }
                Expr::Literal(lit) => match seen.get(lit).map(|(o, _)| o) {
                    None => {
                        seen.insert(lit.clone(), (op, pair));
                    }
                    Some(orig_op) => {
                        seen.insert(lit.clone(), (*orig_op, pair));
                    }
                },
                other => panic!("expected valid map key pattern, got {:?}", &other),
            }
        }

        if !seen.is_empty() {
            // Partition the final group by operator
            let (mut a, mut b) = seen
                .into_values()
                .partition::<Vec<_>, _>(|(op, _pair)| op == &MapOp::Exact);
            if !a.is_empty() {
                groups.push((MapOp::Exact, a.drain(..).map(|(_, p)| p).collect()));
            }
            if !b.is_empty() {
                groups.push((MapOp::Assoc, b.drain(..).map(|(_, p)| p).collect()));
            }
        }

        Ok(groups
            .drain(..)
            .fold((var, pre), |(map, mut pre), (op, pairs)| {
                let span = map.span();
                let (map, mut mpre) = self.force_atomic(map);
                pre.append(&mut mpre);
                (
                    Expr::Map(Map {
                        span,
                        annotations: annotations.clone(),
                        op,
                        var: Box::new(map),
                        pairs,
                    }),
                    pre,
                )
            }))
    }

    /// Force return from body into a list of variables
    fn match_vars(&mut self, expr: Expr) -> (Vec<Var>, Vec<Expr>) {
        match expr {
            Expr::Values(IValues { mut values, .. }) => {
                let mut pre = vec![];
                let mut vars = vec![];
                for expr in values.drain(..) {
                    let (var, mut pre2) = self.force_variable(expr);
                    vars.push(var);
                    pre.append(&mut pre2);
                }
                (vars, pre)
            }
            expr => {
                let (v, pre) = self.force_variable(expr);
                (vec![v], pre)
            }
        }
    }

    /// Transform application
    fn capply(
        &mut self,
        span: SourceSpan,
        annotations: Annotations,
        callee: core::Expr,
        args: Vec<core::Expr>,
        sub: BiMap,
    ) -> Result<(Expr, Vec<Expr>), ExprError> {
        match callee {
            core::Expr::Var(v) if v.arity.is_some() => {
                let f0 = v.name.name;
                let (args, pre) = self.atomic_list(args, sub.clone())?;
                if self.context.labels.contains(&f0) {
                    // This is a goto to a label in a letrec_goto construct
                    let gt = Expr::Goto(Goto {
                        span,
                        annotations: Annotations::default(),
                        label: f0,
                        args,
                    });
                    Ok((gt, pre))
                } else {
                    let arity = v.arity.unwrap();
                    let f1 = sub.get_fsub(f0, arity);
                    let callee = FunctionName::new_local(f1, arity as u8);
                    let call = Expr::Call(Call {
                        span,
                        annotations,
                        callee: Box::new(Expr::Local(Span::new(span, callee))),
                        args,
                        ret: vec![],
                    });
                    Ok((call, pre))
                }
            }
            callee => {
                let (callee, mut pre) = self.variable(callee, sub.clone())?;
                let (args, mut ap) = self.atomic_list(args, sub)?;
                pre.append(&mut ap);
                Ok((
                    Expr::Call(Call {
                        span,
                        annotations,
                        callee: Box::new(callee),
                        args,
                        ret: vec![],
                    }),
                    pre,
                ))
            }
        }
    }

    /// Convert a core expression making sure the result is an atomic
    fn atomic(&mut self, expr: core::Expr, sub: BiMap) -> Result<(Expr, Vec<Expr>), ExprError> {
        let (expr, mut pre) = self.expr(expr, sub)?;
        let (expr, mut pre2) = self.force_atomic(expr);
        pre.append(&mut pre2);
        Ok((expr, pre))
    }

    fn force_atomic(&mut self, expr: Expr) -> (Expr, Vec<Expr>) {
        if expr.is_atomic() {
            (expr, vec![])
        } else {
            let span = expr.span();
            let v = self.context.next_var(Some(span));
            let set = Expr::Set(ISet::new(span, vec![v.clone()], expr, None));
            (Expr::Var(v), vec![set])
        }
    }

    fn atomic_bin(
        &mut self,
        mut segments: Vec<core::Bitstring>,
        sub: BiMap,
    ) -> Result<(Box<Expr>, Vec<Expr>), ExprError> {
        if segments.is_empty() {
            return Ok((Box::new(Expr::BinaryEnd(SourceSpan::default())), vec![]));
        }
        let segment = segments.remove(0);
        let (value, mut pre) = self.atomic(*segment.value, sub.clone())?;
        let (size, mut pre2) = match segment.size {
            None => (None, vec![]),
            Some(sz) => {
                let (sz, szpre) = self.atomic(*sz, sub.clone())?;
                (Some(sz), szpre)
            }
        };
        validate_bin_element_size(size.as_ref(), &segment.annotations)?;
        let (next, mut pre3) = self.atomic_bin(segments, sub)?;

        pre.append(&mut pre2);
        pre.append(&mut pre3);

        Ok((
            Box::new(Expr::BinarySegment(BinarySegment {
                span: segment.span,
                annotations: segment.annotations,
                spec: segment.spec,
                size: size.map(Box::new),
                value: Box::new(value),
                next,
            })),
            pre,
        ))
    }

    fn atomic_list(
        &mut self,
        mut exprs: Vec<core::Expr>,
        sub: BiMap,
    ) -> Result<(Vec<Expr>, Vec<Expr>), ExprError> {
        let mut pre = vec![];
        let mut kexprs = Vec::with_capacity(exprs.len());
        for expr in exprs.drain(..) {
            let (expr, mut pre2) = self.atomic(expr, sub.clone())?;
            kexprs.push(expr);
            pre.append(&mut pre2);
        }
        Ok((kexprs, pre))
    }

    /// Convert a core expression, ensuring the result is a variable
    fn variable(&mut self, expr: core::Expr, sub: BiMap) -> Result<(Expr, Vec<Expr>), ExprError> {
        let (expr, mut pre) = self.expr(expr, sub)?;
        let (v, mut vpre) = self.force_variable(expr);
        pre.append(&mut vpre);
        Ok((Expr::Var(v), pre))
    }

    fn force_variable(&mut self, expr: Expr) -> (Var, Vec<Expr>) {
        match expr {
            Expr::Var(v) => (v, vec![]),
            e => {
                let span = e.span();
                let v = self.context.next_var(Some(span));
                let set = Expr::Set(ISet::new(span, vec![v.clone()], e, None));
                (v, vec![set])
            }
        }
    }

    fn pattern_list(
        &mut self,
        mut patterns: Vec<core::Expr>,
        isub: BiMap,
        osub: BiMap,
    ) -> Result<(Vec<Expr>, BiMap), ExprError> {
        let out = Vec::with_capacity(patterns.len());
        patterns
            .drain(..)
            .try_fold((out, osub), |(mut out, osub0), pat| {
                let (pattern, osub1) = self.pattern(pat, isub.clone(), osub0)?;
                out.push(pattern);
                Ok((out, osub1))
            })
    }

    /// pattern(Cpat, Isub, Osub, State) -> {Kpat,Sub,State}.
    ///  Convert patterns.  Variables shadow so rename variables that are
    ///  already defined.
    ///
    ///  Patterns are complicated by sizes in binaries.  These are pure
    ///  input variables which create no bindings.  We, therefore, need to
    ///  carry around the original substitutions to get the correct
    ///  handling.
    fn pattern(
        &mut self,
        pattern: core::Expr,
        isub: BiMap,
        osub: BiMap,
    ) -> Result<(Expr, BiMap), ExprError> {
        match pattern {
            core::Expr::Var(v) => {
                if self.context.defined.contains(&v.name) {
                    let new = self.context.next_var(Some(v.name.span));
                    let osub = osub.set(Name::from(&v), Name::from(&new));
                    self.context.defined.insert_mut(new.name);
                    Ok((Expr::Var(new), osub))
                } else {
                    self.context.defined.insert_mut(v.name);
                    Ok((Expr::Var(v), osub))
                }
            }
            core::Expr::Literal(lit) => Ok((Expr::Literal(lit), osub)),
            core::Expr::Cons(core::Cons {
                span,
                annotations,
                box head,
                box tail,
            }) => {
                let (head, osub) = self.pattern(head, isub.clone(), osub)?;
                let (tail, osub) = self.pattern(tail, isub, osub)?;
                Ok((
                    Expr::Cons(Cons {
                        span,
                        annotations,
                        head: Box::new(head),
                        tail: Box::new(tail),
                    }),
                    osub,
                ))
            }
            core::Expr::Tuple(core::Tuple {
                span,
                annotations,
                elements,
            }) => {
                let (elements, osub) = self.pattern_list(elements, isub, osub)?;
                Ok((
                    Expr::Tuple(Tuple {
                        span,
                        annotations,
                        elements,
                    }),
                    osub,
                ))
            }
            core::Expr::Map(core::Map {
                span,
                annotations,
                arg,
                pairs,
                ..
            }) => {
                // REVIEW: Is it correct here to handle `arg` as a pattern?
                let (var, osub) = self.pattern(*arg, isub.clone(), osub)?;
                let (pairs, osub) = self.pattern_map_pairs(pairs, isub, osub)?;
                Ok((
                    Expr::Map(Map {
                        span,
                        annotations,
                        var: Box::new(var),
                        op: MapOp::Exact,
                        pairs,
                    }),
                    osub,
                ))
            }
            core::Expr::Binary(core::Binary {
                span,
                annotations,
                segments,
            }) => {
                let (segment, osub) = self.pattern_bin(segments, isub, osub)?;
                Ok((
                    Expr::Binary(Binary {
                        span,
                        annotations,
                        segment,
                    }),
                    osub,
                ))
            }
            core::Expr::Alias(core::Alias {
                span,
                annotations,
                var,
                box pattern,
            }) => {
                let (mut vars, pattern) = flatten_alias(pattern);
                let mut vs = vec![core::Expr::Var(var)];
                vs.extend(vars.drain(..).map(core::Expr::Var));
                let (mut vars, osub) = self.pattern_list(vs, isub.clone(), osub)?;
                let vars = vars.drain(..).map(|v| v.into_var().unwrap()).collect();
                let (pattern, osub) = self.pattern(pattern, isub, osub)?;
                Ok((
                    Expr::Alias(IAlias {
                        span,
                        annotations,
                        vars,
                        pattern: Box::new(pattern),
                    }),
                    osub,
                ))
            }
            invalid => panic!("invalid core expression in pattern: {:?}", &invalid),
        }
    }

    fn pattern_map_pairs(
        &mut self,
        mut pairs: Vec<core::MapPair>,
        isub: BiMap,
        osub: BiMap,
    ) -> Result<(Vec<MapPair>, BiMap), ExprError> {
        // Pattern the pair keys and values as normal
        let mut kpairs = Vec::with_capacity(pairs.len());
        let mut osub1 = osub.clone();
        for core::MapPair {
            op,
            box key,
            box value,
        } in pairs.drain(..)
        {
            assert_eq!(op, MapOp::Exact);
            let (key, _) = self.expr(key, isub.clone())?;
            let (value, osub2) = self.pattern(value, isub.clone(), osub1)?;
            osub1 = osub2;
            kpairs.push(MapPair {
                key: Box::new(key),
                value: Box::new(value),
            });
        }
        // It is later assumed that these keys are term sorted, so we need to sort them here
        kpairs.sort_by(|a, b| a.cmp(b));
        Ok((kpairs, osub1))
    }

    fn pattern_bin(
        &mut self,
        mut segments: Vec<core::Bitstring>,
        isub: BiMap,
        osub: BiMap,
    ) -> Result<(Box<Expr>, BiMap), ExprError> {
        if segments.is_empty() {
            return Ok((Box::new(Expr::BinaryEnd(SourceSpan::default())), osub));
        }
        let segment = segments.remove(0);
        let size = match segment.size {
            None => None,
            Some(sz) => {
                let (sz, _) = self.expr(*sz, isub.clone())?;
                match sz {
                    sz @ Expr::Var(_) => Some(sz),
                    Expr::Literal(Literal {
                        value: Lit::Atom(symbols::All),
                        ..
                    }) => None,
                    sz @ Expr::Literal(_) if sz.is_integer() => Some(sz),
                    // Bad size (coming from an optimization or source code), replace it
                    // with a known atom to avoid accidentally treating it like a real size
                    sz => Some(Expr::Literal(Literal::atom(sz.span(), symbols::BadSize))),
                }
            }
        };
        let (value, osub) = self.pattern(*segment.value, isub.clone(), osub)?;
        let (next, osub) = self.pattern_bin(segments, isub, osub)?;
        let result = self.build_bin_seg(
            segment.span,
            segment.annotations,
            segment.spec,
            size.map(Box::new),
            value,
            next,
        );
        Ok((result, osub))
    }

    /// build_bin_seg(Anno, Size, Unit, Type, Flags, Seg, Next) -> #k_bin_seg{}.
    ///  This function normalizes literal integers with size > 8 and literal
    ///  utf8 segments into integers with size = 8 (and potentially an integer
    ///  with size less than 8 at the end). This is so further optimizations
    ///  have a normalized view of literal integers, allowing us to generate
    ///  more literals and group more clauses. Those integers may be "squeezed"
    ///  later into the largest integer possible.
    fn build_bin_seg(
        &mut self,
        span: SourceSpan,
        annotations: Annotations,
        spec: BinaryEntrySpecifier,
        size: Option<Box<Expr>>,
        value: Expr,
        next: Box<Expr>,
    ) -> Box<Expr> {
        match spec {
            BinaryEntrySpecifier::Integer {
                signed: false,
                endianness: Endianness::Big,
                unit,
            } => match size.as_deref() {
                Some(Expr::Literal(Literal {
                    value: Lit::Integer(Int::Small(sz)),
                    ..
                })) => {
                    let size = (*sz as usize) * unit as usize;
                    match &value {
                        Expr::Literal(Literal {
                            value: Lit::Integer(ref i),
                            ..
                        }) => {
                            if integer_fits_and_is_expandable(i, size) {
                                return build_bin_seg_integer_recur(
                                    span,
                                    annotations,
                                    size,
                                    i.clone(),
                                    next,
                                );
                            }
                        }
                        _ => (),
                    }
                }
                _ => (),
            },
            BinaryEntrySpecifier::Utf8 => match &value {
                Expr::Literal(Literal {
                    value: Lit::Integer(ref i),
                    ..
                }) => {
                    if let Some(c) = i.to_char() {
                        let bits = c.len_utf8() * 8;
                        return build_bin_seg_integer_recur(
                            span,
                            annotations,
                            bits,
                            Int::from(c as u32),
                            next,
                        );
                    }
                }
                _ => (),
            },
            _ => (),
        }
        Box::new(Expr::BinarySegment(BinarySegment {
            span,
            annotations,
            spec,
            size,
            value: Box::new(value),
            next,
        }))
    }
}

#[derive(PartialEq, Eq)]
enum MatchGroupKey {
    Lit(Lit),
    Arity(usize),
    Bin(Option<Box<MatchGroupKey>>, BinaryEntrySpecifier),
    Map(Vec<MapKey>),
    Var(Symbol),
}
impl MatchGroupKey {
    fn from(arg: &Expr, clause: &IClause) -> Self {
        match arg {
            Expr::Literal(Literal { value, .. }) => Self::Lit(value.clone()),
            Expr::Tuple(Tuple { elements, .. }) => Self::Arity(elements.len()),
            Expr::BinarySegment(BinarySegment { size, spec, .. })
            | Expr::BinaryInt(BinarySegment { size, spec, .. }) => match size.as_deref() {
                None => Self::Bin(None, *spec),
                Some(Expr::Var(v)) => {
                    let v1 = clause.isub.get_vsub(v.name());
                    Self::Bin(Some(Box::new(Self::Var(v1))), *spec)
                }
                Some(Expr::Literal(Literal { value, .. })) => {
                    Self::Bin(Some(Box::new(Self::Lit(value.clone()))), *spec)
                }
                _ => unimplemented!(),
            },
            Expr::Map(Map { pairs, .. }) => {
                let mut keys = pairs.iter().map(MapKey::from).collect::<Vec<_>>();
                keys.sort();
                Self::Map(keys)
            }
            Expr::Alias(IAlias { ref pattern, .. }) => MatchGroupKey::from(pattern, clause),
            other => unimplemented!("{:#?}", &other),
        }
    }
}
impl PartialOrd for MatchGroupKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MatchGroupKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::Lit(x), Self::Lit(y)) => x.cmp(y),
            (Self::Lit(x), Self::Arity(y)) => {
                let lit = Lit::Integer(Int::Small(*y as i64));
                x.cmp(&lit)
            }
            (Self::Lit(_), _) => Ordering::Less,
            (Self::Arity(x), Self::Arity(y)) => x.cmp(y),
            (Self::Arity(x), Self::Lit(y)) => {
                let x = Lit::Integer(Int::Small(*x as i64));
                x.cmp(y)
            }
            (_, Self::Lit(_)) => Ordering::Greater,
            (_, Self::Arity(_)) => Ordering::Greater,
            (Self::Var(x), Self::Var(y)) => x.cmp(y),
            (Self::Var(_), _) => Ordering::Less,
            (_, Self::Var(_)) => Ordering::Greater,
            (Self::Bin(xs, xspec), Self::Bin(ys, yspec)) => {
                xs.cmp(ys).then_with(|| xspec.cmp(yspec))
            }
            (Self::Bin(_, _), _) => Ordering::Less,
            (_, Self::Bin(_, _)) => Ordering::Greater,
            (Self::Map(xs), Self::Map(ys)) => xs.cmp(ys),
            (_, Self::Map(_)) => Ordering::Less,
        }
    }
}

fn group_value(
    ty: MatchType,
    vars: Vec<Var>,
    clauses: Vec<IClause>,
) -> Vec<(Vec<Var>, Vec<IClause>)> {
    match ty {
        MatchType::Cons => vec![(vars, clauses)],
        MatchType::Nil => vec![(vars, clauses)],
        MatchType::Binary => vec![(vars, clauses)],
        MatchType::BinaryEnd => vec![(vars, clauses)],
        MatchType::BinarySegment => group_keeping_order(vars, clauses),
        MatchType::BinaryInt => vec![(vars, clauses)],
        MatchType::Map => group_keeping_order(vars, clauses),
        _ => {
            let mut map = group_values(clauses);
            // We must sort the grouped values to ensure consistent order across compilations
            let mut result = vec![];
            while let Some((_, clauses)) = map.pop_first() {
                result.push((vars.clone(), clauses));
            }
            result
        }
    }
}

fn group_values(mut clauses: Vec<IClause>) -> BTreeMap<MatchGroupKey, Vec<IClause>> {
    use std::collections::btree_map::Entry;

    let mut acc = BTreeMap::new();
    for clause in clauses.drain(..) {
        let key = MatchGroupKey::from(clause.arg(), &clause);
        match acc.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(vec![clause]);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().push(clause);
            }
        }
    }
    acc
}

fn group_keeping_order(vars: Vec<Var>, mut clauses: Vec<IClause>) -> Vec<(Vec<Var>, Vec<IClause>)> {
    if clauses.is_empty() {
        return vec![];
    }
    let clause = clauses.remove(0);
    let v1 = MatchGroupKey::from(clause.arg(), &clause);
    let (mut more, rest) = splitwith(clauses, |c| MatchGroupKey::from(c.arg(), c) == v1);
    more.insert(0, clause);
    let group = (vars.clone(), more);
    let mut tail = group_keeping_order(vars, rest);
    tail.insert(0, group);
    tail
}

fn build_bin_seg_integer_recur(
    span: SourceSpan,
    annotations: Annotations,
    bits: usize,
    n: Int,
    next: Box<Expr>,
) -> Box<Expr> {
    if bits > 8 {
        let next_bits = bits - 8;
        let next_value = n.clone() & (((1u64 << next_bits) - 1) as i64);
        let last =
            build_bin_seg_integer_recur(span, annotations.clone(), next_bits, next_value, next);
        build_bin_seg_integer(span, annotations, 8, n >> (next_bits as u32), last)
    } else {
        build_bin_seg_integer(span, annotations, bits, n, next)
    }
}

fn build_bin_seg_integer(
    span: SourceSpan,
    annotations: Annotations,
    bits: usize,
    n: Int,
    next: Box<Expr>,
) -> Box<Expr> {
    let size = Some(Box::new(Expr::Literal(Literal::integer(span, bits))));
    let value = Box::new(Expr::Literal(Literal::integer(span, n)));
    Box::new(Expr::BinarySegment(BinarySegment {
        span,
        annotations,
        spec: BinaryEntrySpecifier::default(),
        size,
        value,
        next,
    }))
}

fn integer_fits_and_is_expandable(i: &Int, size: usize) -> bool {
    if size > EXPAND_MAX_SIZE_SEGMENT {
        return false;
    }
    size as u64 >= i.bits()
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum CallType {
    Error,
    Bif(FunctionName),
    Static(FunctionName),
    Dynamic,
}

fn call_type(module: &core::Expr, function: &core::Expr, args: &[core::Expr]) -> CallType {
    let arity = args.len() as u8;
    match (module.as_atom(), function.as_atom()) {
        (Some(m), Some(f)) => {
            let callee = FunctionName::new(m, f, arity);
            if callee.is_bif() {
                CallType::Bif(callee)
            } else {
                CallType::Static(callee)
            }
        }
        (None, Some(_)) if module.is_var() => CallType::Dynamic,
        (Some(_), None) if function.is_var() => CallType::Dynamic,
        (None, None) if module.is_var() && function.is_var() => CallType::Dynamic,
        _ => CallType::Error,
    }
}

fn same_args(args: &[core::Expr], fargs: &[Var], sub: &BiMap) -> bool {
    if args.len() != fargs.len() {
        return false;
    }
    for (arg, farg) in args.iter().zip(fargs.iter()) {
        match arg {
            core::Expr::Var(ref v) => {
                let vname = Name::from(v);
                let fname = Name::from(farg);
                let vsub = sub.get(vname).unwrap_or(vname);
                if vsub != fname {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

fn make_clist(mut items: Vec<core::Expr>) -> core::Expr {
    items.drain(..).rfold(
        core::Expr::Literal(Literal::nil(SourceSpan::default())),
        |tail, head| {
            let span = head.span();
            core::Expr::Cons(core::Cons::new(span, head, tail))
        },
    )
}

fn flatten_seq(expr: Expr) -> Vec<Expr> {
    match expr {
        Expr::Set(mut iset) => match iset.body.take() {
            None => vec![Expr::Set(iset)],
            Some(box body) => {
                let mut seq = vec![Expr::Set(iset)];
                let mut rest = flatten_seq(body);
                seq.append(&mut rest);
                seq
            }
        },
        expr => vec![expr],
    }
}

fn pre_seq(mut pre: Vec<Expr>, body: Expr) -> Expr {
    if pre.is_empty() {
        return body;
    }
    let expr = pre.remove(0);
    match expr {
        Expr::Set(mut iset) => {
            assert_eq!(iset.body, None);
            iset.body = Some(Box::new(pre_seq(pre, body)));
            return Expr::Set(iset);
        }
        arg => {
            let span = arg.span();
            Expr::Set(ISet {
                span,
                annotations: Annotations::default(),
                vars: vec![],
                arg: Box::new(arg),
                body: Some(Box::new(pre_seq(pre, body))),
            })
        }
    }
}

fn validate_bin_element_size(
    size: Option<&Expr>,
    _annotations: &Annotations,
) -> Result<(), ExprError> {
    match size {
        None | Some(Expr::Var(_)) => Ok(()),
        Some(Expr::Literal(Literal {
            value: Lit::Integer(i),
            ..
        })) if i >= &0 => Ok(()),
        Some(Expr::Literal(Literal {
            value: Lit::Atom(symbols::All),
            ..
        })) => Ok(()),
        Some(expr) => Err(ExprError::BadSegmentSize(expr.span())),
    }
}

fn flatten_alias(expr: core::Expr) -> (Vec<Var>, core::Expr) {
    let mut vars = vec![];
    let mut pattern = expr;
    while let core::Expr::Alias(core::Alias {
        var,
        pattern: box p,
        ..
    }) = pattern
    {
        vars.push(var);
        pattern = p;
    }
    (vars, pattern)
}

// This code implements the algorithm for an optimizing compiler for
// pattern matching given "The Implementation of Functional
// Programming Languages" by Simon Peyton Jones. The code is much
// longer as the meaning of constructors is different from the book.
//
// In Erlang many constructors can have different values, e.g. 'atom'
// or 'integer', whereas in the original algorithm thse would be
// different constructors. Our view makes it easier in later passes to
// handle indexing over each type.
//
// Patterns are complicated by having alias variables.  The form of a
// pattern is Pat | {alias,Pat,[AliasVar]}.  This is hidden by access
// functions to pattern arguments but the code must be aware of it.
//
// The compilation proceeds in two steps:
//
// 1. The patterns in the clauses to converted to lists of kernel
// patterns.  The Core clause is now hybrid, this is easier to work
// with.  Remove clauses with trivially false guards, this simplifies
// later passes.  Add locally defined vars and variable subs to each
// clause for later use.
//
// 2. The pattern matching is optimised.  Variable substitutions are
// added to the VarSub structure and new variables are made visible.
// The guard and body are then converted to Kernel form.
impl<'p> TranslateCore<'p> {
    /// kmatch([Var], [Clause], Sub, State) -> {Kexpr,State}.
    fn kmatch(
        &mut self,
        vars: Vec<Var>,
        clauses: Vec<core::Clause>,
        sub: BiMap,
    ) -> Result<Expr, ExprError> {
        // Convert clauses
        let clauses = self.match_pre(clauses, sub)?;
        self.do_match(vars, clauses, None)
    }

    /// match_pre([Cclause], Sub, State) -> {[Clause],State}.
    ///  Must be careful not to generate new substitutions here now!
    ///  Remove clauses with trivially false guards which will never
    ///  succeed.
    fn match_pre(
        &mut self,
        mut clauses: Vec<core::Clause>,
        sub: BiMap,
    ) -> Result<Vec<IClause>, ExprError> {
        clauses
            .drain(..)
            .map(|clause| {
                let (patterns, osub) =
                    self.pattern_list(clause.patterns, sub.clone(), sub.clone())?;
                Ok(IClause {
                    span: clause.span,
                    annotations: clause.annotations,
                    patterns,
                    guard: clause.guard,
                    body: clause.body,
                    isub: sub.clone(),
                    osub,
                })
            })
            .try_collect()
    }

    /// match([Var], [Clause], Default, State) -> {MatchExpr,State}.
    fn do_match(
        &mut self,
        vars: Vec<Var>,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        if vars.is_empty() {
            return self.match_guard(clauses, default);
        }
        let mut partitions = partition_clauses(clauses);
        let joined = partitions
            .drain(..)
            .try_rfold(default, |default, partition| {
                self.match_varcon(vars.clone(), partition, default)
                    .map(Some)
            })?;
        Ok(joined.unwrap())
    }

    /// match_guard([Clause], Default, State) -> {IfExpr,State}.
    ///  Build a guard to handle guards. A guard *ALWAYS* fails if no
    ///  clause matches, there will be a surrounding 'alt' to catch the
    ///  failure.  Drop redundant cases, i.e. those after a true guard.
    fn match_guard(
        &mut self,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        let (clauses, default) = self.match_guard_1(clauses, default)?;
        Ok(build_alt(build_guard(clauses), default).unwrap())
    }

    fn match_guard_1(
        &mut self,
        mut clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<(Vec<GuardClause>, Option<Expr>), ExprError> {
        if clauses.is_empty() {
            return Ok((vec![], default));
        }
        let clause = clauses.remove(0);
        let span = clause.span;
        if clause.guard.is_none()
            || clause
                .guard
                .as_ref()
                .map(|g| g.is_atom_value(symbols::True))
                .unwrap_or_default()
        {
            // The true clause body becomes the default
            let (body, pre) = self.body(*clause.body, clause.osub.clone())?;
            for clause in clauses.iter() {
                if !clause.is_compiler_generated() {
                    self.diagnostics
                        .diagnostic(Severity::Warning)
                        .with_message("pattern cannot match")
                        .with_primary_label(clause.span, "this clause will never match")
                        .with_secondary_label(span, "it is shadowed by this clause")
                        .emit();
                }
            }
            if let Some(default) = default.as_ref() {
                self.diagnostics
                    .diagnostic(Severity::Warning)
                    .with_message("pattern cannot match")
                    .with_primary_label(default.span(), "this clause will never match")
                    .with_secondary_label(span, "it is shadowed by this clause")
                    .emit();
            }
            Ok((vec![], Some(pre_seq(pre, body))))
        } else {
            let guard = *clause.guard.unwrap();
            let span = guard.span();
            let guard = self.guard(guard, clause.osub.clone())?;
            let (body, pre) = self.body(*clause.body, clause.osub)?;
            let (mut gcs, default) = self.match_guard_1(clauses, default)?;
            let gc = GuardClause {
                span,
                annotations: Annotations::default(),
                guard: Box::new(guard),
                body: Box::new(pre_seq(pre, body)),
            };
            gcs.insert(0, gc);
            Ok((gcs, default))
        }
    }

    /// match_varcon([Var], [Clause], Def, [Var], Sub, State) ->
    ///        {MatchExpr,State}.
    fn match_varcon(
        &mut self,
        vars: Vec<Var>,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        if clauses[0].is_var_clause() {
            self.match_var(vars, clauses, default)
        } else {
            self.match_con(vars, clauses, default)
        }
    }

    /// match_var([Var], [Clause], Def, State) -> {MatchExpr,State}.
    ///  Build a call to "select" from a list of clauses all containing a
    ///  variable as the first argument.  We must rename the variable in
    ///  each clause to be the match variable as these clause will share
    ///  this variable and may have different names for it.  Rename aliases
    ///  as well.
    fn match_var(
        &mut self,
        mut vars: Vec<Var>,
        mut clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        let var = vars.remove(0);
        for clause in clauses.iter_mut() {
            let arg = clause.patterns.remove(0);
            // Get the variables to rename
            let aliases = arg.alias();
            let mut vs = Vec::with_capacity(1 + aliases.len());
            vs.push(arg.as_var().unwrap().clone());
            vs.extend(aliases.iter().cloned());

            // Rename and update substitutions
            let osub = vs.iter().fold(clause.osub.clone(), |sub, v| {
                sub.subst_vsub(Name::from(v), Name::from(&var))
            });
            let isub = vs.iter().fold(clause.isub.clone(), |sub, v| {
                sub.subst_vsub(Name::from(v), Name::from(&var))
            });
            clause.isub = isub;
            clause.osub = osub;
        }
        self.do_match(vars, clauses, default)
    }

    /// match_con(Variables, [Clause], Default, State) -> {SelectExpr,State}.
    ///  Build call to "select" from a list of clauses all containing a
    ///  constructor/constant as first argument.  Group the constructors
    ///  according to type, the order is really irrelevant but tries to be
    ///  smart.
    fn match_con(
        &mut self,
        mut vars: Vec<Var>,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Expr, ExprError> {
        // Extract clauses for different constructors (types).
        let l = vars.clone();
        let u = vars.remove(0);
        let selected = select_types(clauses);
        let mut type_clauses = opt_single_valued(selected);
        let select_clauses = type_clauses
            .drain(..)
            .map(|(ty, clauses)| {
                let values = self.match_value(l.as_slice(), ty, clauses, None)?;
                let span = values[0].span();
                let annotations = values[0].annotations().clone();
                Ok(TypeClause {
                    span,
                    annotations,
                    ty,
                    values,
                })
            })
            .try_collect()?;
        let select = build_select(u, select_clauses);
        let alt = build_alt_1st_no_fail(select, default);
        Ok(alt)
    }

    /// match_value([Var], Con, [Clause], Default, State) -> {SelectExpr,State}.
    ///  At this point all the clauses have the same constructor, we must
    ///  now separate them according to value.
    fn match_value(
        &mut self,
        vars: &[Var],
        ty: MatchType,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<Vec<ValueClause>, ExprError> {
        let (vars, clauses) = partition_intersection(ty, vars.to_vec(), clauses);
        let mut grouped = group_value(ty, vars, clauses);
        let clauses = grouped
            .drain(..)
            .map(|(vars, clauses)| self.match_clause(vars, clauses, default.clone()))
            .try_collect()?;
        Ok(clauses)
    }

    /// match_clause([Var], [Clause], Default, State) -> {Clause,State}.
    ///  At this point all the clauses have the same "value".  Build one
    ///  select clause for this value and continue matching.  Rename
    ///  aliases as well.
    fn match_clause(
        &mut self,
        mut vars: Vec<Var>,
        clauses: Vec<IClause>,
        default: Option<Expr>,
    ) -> Result<ValueClause, ExprError> {
        let var = vars.remove(0);
        let (span, annotations) = clauses
            .first()
            .map(|c| (c.span, c.annotations().clone()))
            .unwrap();
        let (value, mut vs) = self.get_match(get_con(clauses.as_slice()));
        let value = sub_size_var(value, clauses.as_slice());
        let clauses = self.new_clauses(clauses, var);
        let clauses = squeeze_clauses_by_bin_integer_count(clauses);
        vs.append(&mut vars);
        let body = self.do_match(vs, clauses, default)?;
        Ok(ValueClause {
            span,
            annotations,
            value: Box::new(value),
            body: Box::new(body),
        })
    }

    fn get_match(&mut self, expr: &Expr) -> (Expr, Vec<Var>) {
        match expr {
            Expr::Cons(Cons { span, .. }) => {
                let span = *span;
                let [head, tail] = self.context.new_vars(Some(span));
                let h = Expr::Var(head.clone());
                let t = Expr::Var(tail.clone());
                (Expr::Cons(Cons::new(span, h, t)), vec![head, tail])
            }
            Expr::Binary(Binary { span, .. }) => {
                let span = *span;
                let v = self.context.next_var(Some(span));
                (
                    Expr::Binary(Binary::new(span, Expr::Var(v.clone()))),
                    vec![v],
                )
            }
            Expr::BinarySegment(ref segment) if segment.is_all() => {
                let span = segment.span();
                let [value, next] = self.context.new_vars(Some(span));
                let segment = BinarySegment {
                    span,
                    annotations: segment.annotations.clone(),
                    spec: segment.spec.clone(),
                    size: segment.size.clone(),
                    value: Box::new(Expr::Var(value)),
                    next: Box::new(Expr::Var(next.clone())),
                };
                (Expr::BinarySegment(segment), vec![next])
            }
            Expr::BinarySegment(ref segment) => {
                let span = segment.span();
                let [value, next] = self.context.new_vars(Some(span));
                let segment = BinarySegment {
                    span,
                    annotations: segment.annotations.clone(),
                    spec: segment.spec.clone(),
                    size: segment.size.clone(),
                    value: Box::new(Expr::Var(value.clone())),
                    next: Box::new(Expr::Var(next.clone())),
                };
                (Expr::BinarySegment(segment), vec![value, next])
            }
            Expr::BinaryInt(ref segment) => {
                let span = segment.span();
                let next = self.context.next_var(Some(span));
                let segment = BinarySegment {
                    span,
                    annotations: segment.annotations.clone(),
                    spec: segment.spec.clone(),
                    size: segment.size.clone(),
                    value: segment.value.clone(),
                    next: Box::new(Expr::Var(next.clone())),
                };
                (Expr::BinaryInt(segment), vec![next])
            }
            Expr::Tuple(Tuple { span, elements, .. }) => {
                let span = *span;
                let vars = self.context.n_vars(elements.len(), Some(span));
                let elements = vars.iter().cloned().map(Expr::Var).collect();
                (Expr::Tuple(Tuple::new(span, elements)), vars)
            }
            Expr::Map(Map { span, pairs, .. }) => {
                let span = *span;
                let vars = self.context.n_vars(pairs.len(), Some(span));
                let pairs = pairs
                    .iter()
                    .zip(vars.iter().cloned())
                    .map(|(pair, v)| MapPair {
                        key: pair.key.clone(),
                        value: Box::new(Expr::Var(v)),
                    })
                    .collect();
                let empty = Box::new(Expr::Literal(Literal::map(span, Default::default())));
                (
                    Expr::Map(Map {
                        span,
                        annotations: Annotations::default(),
                        op: MapOp::Exact,
                        var: empty,
                        pairs,
                    }),
                    vars,
                )
            }
            expr => (expr.clone(), vec![]),
        }
    }

    fn new_clauses(&mut self, mut clauses: Vec<IClause>, var: Var) -> Vec<IClause> {
        clauses
            .drain(..)
            .map(|mut clause| {
                let arg = clause.patterns.remove(0);
                let (osub, isub) = {
                    let vs = arg.alias();
                    let osub = vs.iter().fold(clause.osub.clone(), |osub, v| {
                        osub.subst_vsub(Name::from(v), Name::from(&var))
                    });
                    let isub = vs.iter().fold(clause.isub.clone(), |isub, v| {
                        isub.subst_vsub(Name::from(v), Name::from(&var))
                    });
                    (osub, isub)
                };
                let head = match arg.into_arg() {
                    Expr::Cons(Cons {
                        box head, box tail, ..
                    }) => {
                        let mut hs = Vec::with_capacity(2 + clause.patterns.len());
                        hs.push(head);
                        hs.push(tail);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::Tuple(Tuple { mut elements, .. }) => {
                        let mut hs = Vec::with_capacity(elements.len() + clause.patterns.len());
                        hs.append(&mut elements);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::Binary(Binary {
                        segment: box segment,
                        ..
                    }) => {
                        let mut hs = Vec::with_capacity(1 + clause.patterns.len());
                        hs.push(segment);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::BinarySegment(segment) if segment.is_all() => {
                        let mut hs = Vec::with_capacity(1 + clause.patterns.len());
                        hs.push(*segment.value);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::BinarySegment(segment) => {
                        let mut hs = Vec::with_capacity(2 + clause.patterns.len());
                        hs.push(*segment.value);
                        hs.push(*segment.next);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::BinaryInt(segment) => {
                        let mut hs = Vec::with_capacity(1 + clause.patterns.len());
                        hs.push(*segment.next);
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    Expr::Map(Map { mut pairs, .. }) => {
                        let mut hs = Vec::with_capacity(pairs.len() + clause.patterns.len());
                        hs.extend(pairs.drain(..).map(|p| *p.value));
                        hs.append(&mut clause.patterns);
                        hs
                    }
                    _ => clause.patterns,
                };

                IClause {
                    span: clause.span,
                    annotations: clause.annotations,
                    isub,
                    osub,
                    patterns: head,
                    guard: clause.guard,
                    body: clause.body,
                }
            })
            .collect()
    }
}

fn sub_size_var(mut expr: Expr, clauses: &[IClause]) -> Expr {
    if let Expr::BinarySegment(ref mut segment) = &mut expr {
        let size_var = segment
            .size
            .as_deref()
            .and_then(|sz| sz.as_var())
            .map(|v| (v.span(), v.annotations.clone(), Name::from(v)));
        if let Some((span, annotations, sz)) = size_var {
            if let Some(clause) = clauses.first() {
                let name = clause.isub.get(sz).unwrap_or(sz);
                let size = match name {
                    Name::Var(s) => Expr::Var(Var {
                        annotations,
                        name: Ident::new(s, span),
                        arity: None,
                    }),
                    Name::Fun(s, arity) => Expr::Var(Var {
                        annotations,
                        name: Ident::new(s, span),
                        arity: Some(arity),
                    }),
                };
                segment.size.replace(Box::new(size));
            }
        }
    }

    expr
}

fn select_types(mut clauses: Vec<IClause>) -> Vec<(MatchType, Vec<IClause>)> {
    use std::collections::btree_map::Entry;

    let acc: BTreeMap<MatchType, Vec<IClause>> = BTreeMap::new();
    let mut acc = clauses.drain(..).fold(acc, |mut acc, mut clause| {
        expand_pat_lit_clause(&mut clause);
        let ty = clause.match_type();
        match acc.entry(ty) {
            Entry::Vacant(entry) => {
                entry.insert(vec![clause]);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().push(clause);
            }
        }
        acc
    });

    let mut result = Vec::with_capacity(acc.len());
    while let Some((t, cs)) = acc.pop_first() {
        let cs = match t {
            MatchType::BinarySegment => {
                let mut grouped = handle_bin_con(cs);
                result.append(&mut grouped);
                continue;
            }
            _ => cs,
        };
        result.push((t, cs));
    }
    result
}

/// handle_bin_con([Clause]) -> [{Type,[Clause]}].
///  Handle clauses for the k_bin_seg constructor.  As k_bin_seg
///  matching can overlap, the k_bin_seg constructors cannot be
///  reordered, only grouped.
fn handle_bin_con(clauses: Vec<IClause>) -> Vec<(MatchType, Vec<IClause>)> {
    if is_select_bin_int_possible(clauses.as_slice()) {
        // The usual way to match literals is to first extract the
        // value to a register, and then compare the register to the
        // literal value. Extracting the value is good if we need
        // compare it more than once.
        //
        // But we would like to combine the extracting and the
        // comparing into a single instruction if we know that
        // a binary segment must contain specific integer value
        // or the matching will fail, like in this example:
        //
        // <<42:8,...>> ->
        // <<42:8,...>> ->
        // .
        // .
        // .
        // <<42:8,...>> ->
        // <<>> ->
        //
        // The first segment must either contain the integer 42
        // or the binary must end for the match to succeed.
        //
        // The way we do is to replace the generic #k_bin_seg{}
        // record with a #k_bin_int{} record if all clauses will
        // select the same literal integer (except for one or more
        // clauses that will end the binary).
        let (mut bin_segs, bin_end) = splitwith(clauses, |clause| {
            clause.match_type() == MatchType::BinarySegment
        });

        // Swap all BinarySegment exprs with BinaryInt exprs
        let mut dummy = Expr::Literal(Literal::nil(SourceSpan::default()));
        for clause in bin_segs.iter_mut() {
            let pattern = clause.patterns.get_mut(0).unwrap();
            let Expr::BinarySegment(mut bs) = mem::replace(pattern, dummy) else { panic!("expected binary segment pattern") };
            // To make lowering easier, force the size to always be set, i..e. if the
            // size value is None, set it to the default of 8
            if bs.size.is_none() {
                let span = bs.span;
                bs.size
                    .replace(Box::new(Expr::Literal(Literal::integer(span, 8))));
            }
            dummy = mem::replace(pattern, Expr::BinaryInt(bs));
        }

        let mut bin_segs = vec![(MatchType::BinaryInt, bin_segs)];
        if !bin_end.is_empty() {
            bin_segs.push((MatchType::BinaryEnd, bin_end));
        }
        bin_segs
    } else {
        handle_bin_con_not_possible(clauses)
    }
}

fn handle_bin_con_not_possible(clauses: Vec<IClause>) -> Vec<(MatchType, Vec<IClause>)> {
    if clauses.is_empty() {
        return vec![];
    }
    let con = clauses[0].match_type();
    let (more, rest) = splitwith(clauses, |clause| clause.match_type() == con);
    let mut result = vec![(con, more)];
    let mut rest = handle_bin_con_not_possible(rest);
    result.append(&mut rest);
    result
}

fn is_select_bin_int_possible(clauses: &[IClause]) -> bool {
    if clauses.is_empty() {
        return false;
    }
    // Use the first clause to determine how to check the rest
    let match_bits;
    let match_size;
    let match_value;
    let match_signed;
    let match_endianness;
    {
        match clauses[0].patterns.first().unwrap() {
            Expr::BinarySegment(BinarySegment {
                spec:
                    BinaryEntrySpecifier::Integer {
                        unit,
                        signed,
                        endianness,
                    },
                size,
                value:
                    box Expr::Literal(Literal {
                        value: Lit::Integer(ref i),
                        ..
                    }),
                ..
            }) => {
                match_size = match size.as_deref() {
                    None => 8,
                    Some(Expr::Literal(Literal {
                        value: Lit::Integer(ref i),
                        ..
                    })) => match i.to_usize() {
                        None => return false,
                        Some(sz) => sz,
                    },
                    _ => return false,
                };
                match_bits = (*unit as usize) * match_size;
                match_signed = *signed;
                match_endianness = *endianness;
                match_value = i.clone();
                // Expands the code size too much
                if match_bits > EXPAND_MAX_SIZE_SEGMENT {
                    return false;
                }
                // Can't know the native endianness at this point
                if match_endianness == Endianness::Native {
                    return false;
                }
                if !select_match_possible(match_size, &match_value, match_signed, match_endianness)
                {
                    return false;
                }
            }
            _ => return false,
        }
    }

    for clause in clauses.iter().skip(1) {
        match clause.patterns.first().unwrap() {
            Expr::BinarySegment(BinarySegment {
                spec:
                    BinaryEntrySpecifier::Integer {
                        unit,
                        signed,
                        endianness,
                    },
                size,
                value:
                    box Expr::Literal(Literal {
                        value: Lit::Integer(ref i),
                        ..
                    }),
                ..
            }) => {
                if *signed != match_signed || *endianness != match_endianness || i != &match_value {
                    return false;
                }
                let size = match size.as_deref() {
                    None => 8,
                    Some(Expr::Literal(Literal {
                        value: Lit::Integer(ref i),
                        ..
                    })) => match i.to_usize() {
                        None => return false,
                        Some(sz) => sz,
                    },
                    _ => return false,
                };
                let bits = (*unit as usize) * size;
                if bits != match_bits {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

/// Returns true if roundtripping `value` through the given encoding would succeed
fn select_match_possible(size: usize, value: &Int, signed: bool, endianness: Endianness) -> bool {
    // Make sure there is enough available bits to hold the value
    let needs_bits = value.bits();
    let available_bits = size as u64;
    if needs_bits > available_bits {
        return false;
    }

    // Encode `value` into a binary
    let mut bv = BitVec::new();
    match value {
        Int::Small(i) if signed => {
            bv.push_ap_number(*i, size, endianness);
        }
        Int::Small(i) => {
            bv.push_ap_number(*i as u64, size, endianness);
        }
        Int::Big(ref i) => {
            bv.push_ap_bigint(i, size, signed, endianness);
        }
    }

    // Match an integer using the specified encoding, and make sure it equals the input value
    let mut matcher = bv.matcher();
    match value {
        Int::Small(i) if signed => {
            let parsed: Option<i64> = matcher.read_ap_number(size, endianness);
            match parsed {
                None => false,
                Some(p) => p.eq(i),
            }
        }
        Int::Small(i) => {
            let parsed: Option<u64> = matcher.read_ap_number(size, endianness);
            match parsed {
                None => false,
                Some(p) => p == (*i as u64),
            }
        }
        Int::Big(ref i) => {
            let parsed = matcher.read_bigint(size, signed, endianness);
            match parsed {
                None => false,
                Some(p) => p.eq(i),
            }
        }
    }
}

/// partition([Clause]) -> [[Clause]].
///  Partition a list of clauses into groups which either contain
///  clauses with a variable first argument, or with a "constructor".
fn partition_clauses(clauses: Vec<IClause>) -> Vec<Vec<IClause>> {
    if clauses.is_empty() {
        return vec![];
    }
    let v1 = clauses[0].is_var_clause() == true;
    let (more, rest) = splitwith(clauses, |c| c.is_var_clause() == v1);
    let mut cs = partition_clauses(rest);
    cs.insert(0, more);
    cs
}

fn expand_pat_lit_clause(clause: &mut IClause) {
    match clause.patterns.get_mut(0).unwrap() {
        Expr::Alias(ref mut alias) if alias.pattern.is_literal() => {
            expand_pat_lit(alias.pattern.as_mut());
        }
        lit @ Expr::Literal(_) => {
            expand_pat_lit(lit);
        }
        _ => (),
    }
}

fn expand_pat_lit(pattern: &mut Expr) {
    let pat = match pattern {
        Expr::Literal(Literal {
            span,
            annotations,
            value,
        }) => match value {
            Lit::Cons(head, tail) => Expr::Cons(Cons {
                span: *span,
                annotations: annotations.clone(),
                head: Box::new(Expr::Literal(head.as_ref().clone())),
                tail: Box::new(Expr::Literal(tail.as_ref().clone())),
            }),
            Lit::Tuple(t) => Expr::Tuple(Tuple {
                span: *span,
                annotations: annotations.clone(),
                elements: t.iter().cloned().map(Expr::Literal).collect(),
            }),
            _ => return,
        },
        _ => return,
    };
    *pattern = pat;
}

/// opt_single_valued([{Type,Clauses}]) -> [{Type,Clauses}].
///  If a type only has one clause and if the pattern is a complex
///  literal, the matching can be done more efficiently by directly
///  comparing with the literal (that is especially true for binaries).
///
///  It is important not to do this transformation for atomic literals
///  (such as `[]`), since that would cause the test for an empty list
///  to be executed before the test for a nonempty list.
fn opt_single_valued(
    mut tclauses: Vec<(MatchType, Vec<IClause>)>,
) -> Vec<(MatchType, Vec<IClause>)> {
    let mut lcs = vec![];
    let mut tcs = vec![];

    for (t, mut clauses) in tclauses.drain(..) {
        if clauses.len() == 1 {
            let mut clause = clauses.pop().unwrap();
            if clause.patterns[0].is_literal() {
                // This is an atomic literal
                tcs.push((t, vec![clause]));
                continue;
            }

            let pattern = clause.patterns[0].clone();
            match combine_lit_pat(pattern) {
                Ok(pattern) => {
                    let _ = mem::replace(&mut clause.patterns[0], pattern);
                    lcs.push(clause);
                    continue;
                }
                Err(_) => {
                    // Not possible
                    tcs.push((t, vec![clause]));
                    continue;
                }
            }
        } else {
            tcs.push((t, clauses));
        }
    }

    if lcs.is_empty() {
        tcs
    } else {
        let literals = (MatchType::Literal, lcs);
        // Test the literals as early as possible.
        match tcs.first().map(|(t, _)| *t).unwrap() {
            MatchType::Binary => {
                // The delayed creation of sub binaries requires
                // bs_start_match2 to be the first instruction in the
                // function
                tcs.insert(1, literals);
                tcs
            }
            _ => {
                tcs.insert(0, literals);
                tcs
            }
        }
    }
}

fn combine_lit_pat(pattern: Expr) -> Result<Expr, ()> {
    match pattern {
        Expr::Alias(IAlias {
            span,
            annotations,
            vars,
            box pattern,
        }) => {
            let pattern = combine_lit_pat(pattern)?;
            Ok(Expr::Alias(IAlias {
                span,
                annotations,
                vars,
                pattern: Box::new(pattern),
            }))
        }
        Expr::Literal(_) => {
            // This is an atomic literal. Rewriting would be a pessimization, especially for nil
            Err(())
        }
        pattern => {
            let lit = do_combine_lit_pat(pattern)?;
            Ok(Expr::Literal(lit))
        }
    }
}

fn do_combine_lit_pat(pattern: Expr) -> Result<Literal, ()> {
    match pattern {
        Expr::Literal(lit) => Ok(lit),
        Expr::Binary(Binary {
            span,
            annotations,
            segment,
        }) => {
            let mut bv = BitVec::new();
            combine_bin_segs(*segment, &mut bv)?;
            Ok(Literal {
                span,
                annotations,
                value: Lit::Binary(bv),
            })
        }
        Expr::Cons(Cons {
            span,
            annotations,
            box head,
            box tail,
        }) => {
            let head = do_combine_lit_pat(head)?;
            let tail = do_combine_lit_pat(tail)?;
            Ok(Literal {
                span,
                annotations,
                value: Lit::Cons(Box::new(head), Box::new(tail)),
            })
        }
        Expr::Tuple(Tuple {
            span,
            annotations,
            mut elements,
        }) => {
            let elements = elements.drain(..).map(do_combine_lit_pat).try_collect()?;
            Ok(Literal {
                span,
                annotations,
                value: Lit::Tuple(elements),
            })
        }
        _ => Err(()),
    }
}

fn combine_bin_segs(segment: Expr, bin: &mut BitVec) -> Result<(), ()> {
    match segment {
        Expr::BinaryEnd(_) => Err(()),
        Expr::BinarySegment(BinarySegment {
            spec:
                BinaryEntrySpecifier::Integer {
                    unit: 1,
                    signed: false,
                    endianness: Endianness::Big,
                },
            size,
            value,
            next,
            ..
        }) => match size.as_deref() {
            None => match *value {
                Expr::Literal(Literal {
                    value: Lit::Integer(i),
                    ..
                }) if i >= 0 && i <= 255 => {
                    let byte = i.try_into().map_err(|_| ())?;
                    bin.push_byte(byte);
                    combine_bin_segs(*next, bin)
                }
                _ => Err(()),
            },
            Some(Expr::Literal(Literal {
                value: Lit::Integer(i),
                ..
            })) if i == &8 => match *value {
                Expr::Literal(Literal {
                    value: Lit::Integer(i),
                    ..
                }) if i >= 0 && i <= 255 => {
                    let byte = i.try_into().map_err(|_| ())?;
                    bin.push_byte(byte);
                    combine_bin_segs(*next, bin)
                }
                _ => Err(()),
            },
            _ => Err(()),
        },
        _other => Err(()),
    }
}

/// partition_intersection(Type, Us, [Clause], State) -> {Us,Cs,State}.
///  Partitions a map into two maps with the most common keys to the
///  first map.
///
///      case <M> of
///          <#{a,b}>
///          <#{a,c}>
///          <#{a}>
///      end
///
///  becomes
///
///      case <M,M> of
///          <#{a}, #{b}>
///          <#{a}, #{c}>
///          <#{a}, #{ }>
///      end
///
///  The intention is to group as many keys together as possible and
///  thus reduce the number of lookups to that key.
fn partition_intersection(
    ty: MatchType,
    mut vars: Vec<Var>,
    mut clauses: Vec<IClause>,
) -> (Vec<Var>, Vec<IClause>) {
    match ty {
        MatchType::Map => {
            let patterns = clauses
                .iter()
                .map(|clause| match clause.arg().arg() {
                    Expr::Map(Map { pairs, .. }) => pairs.iter().map(MapKey::from).collect(),
                    arg => panic!("expected clause arg to be a map expression, got: {:?}", arg),
                })
                .collect();
            match find_key_intersection(patterns) {
                None => (vars, clauses),
                Some(keys) => {
                    for clause in clauses.iter_mut() {
                        let arg = clause.patterns.remove(0);
                        let (arg1, arg2) = partition_keys(arg, &keys);
                        let mut patterns = Vec::with_capacity(2 + clause.patterns.len());
                        patterns.push(arg1);
                        patterns.push(arg2);
                        patterns.append(&mut clause.patterns);
                        clause.patterns = patterns;
                    }
                    // Duplicate the var
                    let u = vars[0].clone();
                    vars.insert(0, u);
                    (vars, clauses)
                }
            }
        }
        _ => (vars, clauses),
    }
}

fn find_key_intersection(patterns: Vec<BTreeSet<MapKey>>) -> Option<BTreeSet<MapKey>> {
    if patterns.is_empty() {
        return None;
    }

    // Get the intersection of all the sets
    let base = patterns.first().unwrap().clone();
    let intersection: BTreeSet<MapKey> = patterns
        .iter()
        .skip(1)
        .fold(base, |acc, set| acc.intersection(set).cloned().collect());
    if intersection.is_empty() {
        None
    } else {
        // If all the clauses test the same keys, partitioning can only
        // make the generated code worse
        if patterns.iter().all(|set| set.eq(&intersection)) {
            None
        } else {
            Some(intersection)
        }
    }
}

fn partition_keys(arg: Expr, keys: &BTreeSet<MapKey>) -> (Expr, Expr) {
    match arg {
        Expr::Map(Map {
            span,
            annotations,
            op,
            var,
            mut pairs,
        }) => {
            let (ps1, ps2) = pairs.drain(..).partition(|pair| {
                let key = MapKey::from(pair.key.as_ref());
                keys.contains(&key)
            });
            let arg1 = Expr::Map(Map {
                span,
                annotations: annotations.clone(),
                op,
                var: var.clone(),
                pairs: ps1,
            });
            let arg2 = Expr::Map(Map {
                span,
                annotations,
                op,
                var,
                pairs: ps2,
            });
            (arg1, arg2)
        }
        Expr::Alias(IAlias {
            span,
            annotations,
            vars,
            pattern,
        }) => {
            // Only alias one of them
            let (map1, map2) = partition_keys(*pattern, keys);
            let alias = Expr::Alias(IAlias {
                span,
                annotations,
                vars,
                pattern: Box::new(map2),
            });
            (map1, alias)
        }
        other => panic!(
            "unexpected argument to partition_keys, expected map or alias expr, got: {:?}",
            &other
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum MapKey {
    Literal(Lit),
    Var(Symbol),
}
impl From<&MapPair> for MapKey {
    fn from(pair: &MapPair) -> Self {
        pair.key.as_ref().into()
    }
}
impl From<&Expr> for MapKey {
    fn from(expr: &Expr) -> Self {
        match expr {
            Expr::Var(v) => Self::Var(v.name()),
            Expr::Literal(Literal { value, .. }) => Self::Literal(value.clone()),
            _ => unimplemented!(),
        }
    }
}

fn get_con(clauses: &[IClause]) -> &Expr {
    clauses.first().unwrap().arg().arg()
}

fn build_guard(clauses: Vec<GuardClause>) -> Option<Expr> {
    if clauses.is_empty() {
        return None;
    }
    Some(Expr::Guard(Guard {
        span: SourceSpan::default(),
        annotations: Annotations::default(),
        clauses,
    }))
}

/// Build an alt, attempt some simple optimization
fn build_alt(first: Option<Expr>, other: Option<Expr>) -> Option<Expr> {
    if first.is_none() {
        return other;
    }
    Some(build_alt_1st_no_fail(first.unwrap(), other))
}

fn build_alt_1st_no_fail(first: Expr, other: Option<Expr>) -> Expr {
    if other.is_none() {
        return first;
    }
    Expr::Alt(Alt {
        span: first.span(),
        annotations: first.annotations().clone(),
        first: Box::new(first),
        then: Box::new(other.unwrap()),
    })
}

fn build_select(var: Var, types: Vec<TypeClause>) -> Expr {
    let first = types.first().unwrap();
    let annotations = first.annotations().clone();
    let span = first.span();
    Expr::Select(Select {
        span,
        annotations,
        var,
        types,
    })
}

/// Build a match expresssion if there is a match
fn build_match(expr: Expr) -> Expr {
    match expr {
        expr @ (Expr::Alt(_) | Expr::Select(_) | Expr::Guard(_)) => {
            let span = expr.span();
            let annotations = expr.annotations().clone();
            Expr::Match(Match {
                span,
                annotations,
                body: Box::new(expr),
                ret: vec![],
            })
        }
        expr => expr,
    }
}

fn squeeze_clauses_by_bin_integer_count(clauses: Vec<IClause>) -> Vec<IClause> {
    // TODO
    clauses
}

/// Partitions the given vector into two vectors based on the predicate.
///
/// This is essentially equivalent to `Vec::split_off/1`, but where the
/// partition point is determined by the predicate rather than given explicitly.
fn splitwith<T, F>(mut items: Vec<T>, predicate: F) -> (Vec<T>, Vec<T>)
where
    F: FnMut(&T) -> bool,
{
    let index = items.partition_point(predicate);
    let rest = items.split_off(index);

    (items, rest)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Brk {
    Return,
    Break(Vec<Expr>),
}
impl Brk {
    pub fn is_break(&self) -> bool {
        match self {
            Self::Break(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_return(&self) -> bool {
        !self.is_break()
    }

    pub fn ret(&self) -> &[Expr] {
        match self {
            Self::Break(rs) => rs.as_slice(),
            Self::Return => &[],
        }
    }

    pub fn into_ret(self) -> Vec<Expr> {
        match self {
            Self::Break(rs) => rs,
            Self::Return => vec![],
        }
    }
}

impl<'p> TranslateCore<'p> {
    /// ubody(Expr, Break, State) -> {Expr,[UsedVar],State}.
    ///  Tag the body sequence with its used variables.  These bodies
    ///  either end with a #k_break{}, or with #k_return{} or an expression
    ///  which itself can return, #k_enter{}, #k_match{} ... .
    fn ubody(&mut self, body: Expr, brk: Brk) -> Result<(Expr, RedBlackTreeSet<Ident>), ExprError> {
        match body {
            // A letrec should never be last
            Expr::Set(ISet {
                vars,
                arg: box Expr::LetRec(letr),
                body: Some(box body),
                ..
            }) if vars.is_empty() => {
                self.iletrec_funs(letr)?;
                self.ubody(body, brk)
            }
            Expr::Set(ISet {
                vars,
                arg: box Expr::Literal(_),
                body: Some(box body),
                ..
            }) if vars.is_empty() => self.ubody(body, brk),
            Expr::Set(ISet {
                span,
                annotations,
                vars,
                box arg,
                body: Some(box body),
            }) => {
                let ns = vars.iter().map(|v| v.name).collect();
                let vs = vars.iter().cloned().map(Expr::Var).collect();
                let (arg, au) = self.uexpr(arg, Brk::Break(vs))?;
                let (body, bu) = self.ubody(body, brk)?;
                let used = sets::union(au, sets::subtract(bu, ns)); // used external vars
                Ok((
                    Expr::Seq(Seq {
                        span,
                        annotations,
                        arg: Box::new(arg),
                        body: Box::new(body),
                    }),
                    used,
                ))
            }
            Expr::Values(IValues {
                annotations,
                values,
                ..
            }) => match brk {
                Brk::Return => {
                    let span = values[0].span();
                    let au = lit_list_vars(values.as_slice());
                    Ok((
                        Expr::Return(Return {
                            span,
                            annotations,
                            args: values,
                        }),
                        au,
                    ))
                }
                Brk::Break(_) => {
                    let span = values.first().map(|a| a.span()).unwrap_or_default();
                    let au = lit_list_vars(values.as_slice());
                    Ok((
                        Expr::Break(Break {
                            span,
                            annotations,
                            args: values,
                        }),
                        au,
                    ))
                }
            },
            goto @ Expr::Goto(_) => Ok((goto, rbt_set![])),

            // Enterable expressions need no trailing return
            expr => match brk {
                Brk::Return if expr.is_enter_expr() => self.uexpr(expr, Brk::Return),
                Brk::Return => {
                    let (ea, pa) = self.force_atomic(expr);
                    self.ubody(pre_seq(pa, Expr::Values(kvalues!(ea))), Brk::Return)
                }
                Brk::Break(args) if args.len() == 1 => {
                    let (ea, pa) = self.force_atomic(expr);
                    self.ubody(pre_seq(pa, Expr::Values(kvalues!(ea))), Brk::Break(args))
                }
                Brk::Break(args) => {
                    let span = args.first().map(|a| a.span());
                    let vars = self.context.n_vars(args.len(), span);
                    let values = vars.iter().cloned().map(Expr::Var).collect();
                    let set = Expr::Set(ISet {
                        span: span.unwrap_or_default(),
                        annotations: Annotations::default(),
                        vars,
                        arg: Box::new(expr),
                        body: None,
                    });
                    let seq = pre_seq(vec![set], Expr::Values(IValues::new(values)));
                    self.ubody(seq, Brk::Break(args))
                }
            },
        }
    }

    fn iletrec_funs(&mut self, mut lr: ILetRec) -> Result<(), ExprError> {
        // Use union of all free variables.
        // First just work out free variables for all functions.
        let free = lr.defs.iter().try_fold(rbt_set![], |free, (_, ref fun)| {
            let ns = fun.vars.iter().map(|v| v.name).collect();
            let fbu = self.ubody_used_vars(Expr::Fun(fun.clone()))?;
            Ok(sets::union(sets::subtract(fbu, ns), free))
        })?;
        let free_vars = free
            .iter()
            .copied()
            .map(|id| Var::new(id))
            .collect::<Vec<_>>();
        for (name, fun) in lr.defs.iter() {
            self.store_free(
                name.name(),
                fun.vars.len(),
                free_vars.iter().cloned().map(Expr::Var).collect(),
            );
        }
        // Now regenerate local functions to use free variable information.
        if self.context.ignore_funs {
            // Optimization: The ultimate caller is only interested in the used
            // variables, not the updated state. Makes a difference if there are
            // nested letrecs.
            Ok(())
        } else {
            // We perform a transformation here for funs that will help later during
            // code generation, and forms the calling convention for closures (note
            // that this does not apply to "empty" closures, i.e. those with no free
            // variables). The transformation works like this:
            //
            // 1. Rewrite the function signature to expect a closure as an extra trailing argument
            // 2. Inject the unpack_env primop in the function entry for each free variable to
            // extract it from the closure argument
            for (
                name,
                IFun {
                    span,
                    mut annotations,
                    mut vars,
                    box body,
                },
            ) in lr.defs.drain(..)
            {
                // Perform the usual rewrite, then wrap the body to contain the unpack_env
                // instructions
                let (body, _) = self.ubody(body, Brk::Return)?;
                let (name, body) = if free_vars.is_empty() {
                    let name = FunctionName::new_local(name.name(), vars.len() as u8);
                    (name, body)
                } else {
                    let name = FunctionName::new_local(name.name(), (vars.len() + 1) as u8);
                    let closure_var = self.context.next_var(Some(span));
                    vars.push(closure_var.clone());
                    annotations.set(symbols::Closure);
                    (
                        name,
                        self.unpack_closure_body(span, body, closure_var, free_vars.clone()),
                    )
                };
                let function = make_function(span, annotations, name, vars, body);
                self.context.funs.push(function);
            }
            Ok(())
        }
    }

    fn unpack_closure_body(
        &mut self,
        span: SourceSpan,
        body: Expr,
        closure: Var,
        mut free: Vec<Var>,
    ) -> Expr {
        assert_ne!(free.len(), 0);

        // We create a chained series of calls to unpack_env/2, in reverse, with appropriate rets
        // for each free variable This will result in the correct sequence of instructions
        // when lowered
        let env_arity = free.len();
        let mut body = body;
        for i in (0..env_arity).rev() {
            let closure = Expr::Var(closure.clone());
            let index = Expr::Literal(Literal::integer(span, i));
            let unpack_env2 = FunctionName::new(symbols::Erlang, symbols::UnpackEnv, 2);
            let mut bif = Bif::new(span, unpack_env2, vec![closure, index]);
            bif.ret.push(Expr::Var(free.pop().unwrap()));
            let seq = Expr::Seq(Seq {
                span,
                annotations: Annotations::default_compiler_generated(),
                arg: Box::new(Expr::Bif(bif)),
                body: Box::new(body),
            });
            body = seq;
        }
        body
    }

    /// uexpr(Expr, Break, State) -> {Expr,[UsedVar],State}.
    ///  Calculate the used variables for an expression.
    ///  Break = return | {break,[RetVar]}.
    fn uexpr(&mut self, expr: Expr, brk: Brk) -> Result<(Expr, RedBlackTreeSet<Ident>), ExprError> {
        match expr {
            Expr::Test(test) if brk.is_break() => {
                // Sanity check
                assert_eq!(brk.ret(), &[]);
                let used = lit_list_vars(test.args.as_slice());
                Ok((Expr::Test(test), used))
            }
            Expr::Set(ISet {
                span,
                annotations,
                vars,
                box arg,
                body: Some(box body),
            }) if brk.is_break() => {
                let ns = vars.iter().map(|v| v.name).collect();
                let vars = vars.iter().cloned().map(Expr::Var).collect();
                let (e1, eu) = self.uexpr(arg, Brk::Break(vars))?;
                let (b1, bu) = self.uexpr(body, brk)?;
                let used = sets::union(eu, sets::subtract(bu, ns));
                Ok((
                    Expr::Seq(Seq {
                        span,
                        annotations,
                        arg: Box::new(e1),
                        body: Box::new(b1),
                    }),
                    used,
                ))
            }
            Expr::If(If {
                span,
                annotations,
                cond,
                box then_body,
                box else_body,
                ..
            }) => {
                let ret = brk.ret().to_vec();
                let cu = lit_vars(cond.as_ref());
                let (then_body, tbu) = self.ubody(then_body, brk.clone())?;
                let (else_body, ebu) = self.ubody(else_body, brk)?;
                let used = sets::union(cu, sets::union(tbu, ebu));
                Ok((
                    Expr::If(If {
                        span,
                        annotations,
                        cond,
                        then_body: Box::new(then_body),
                        else_body: Box::new(else_body),
                        ret,
                    }),
                    used,
                ))
            }
            Expr::Call(mut call) if call.callee.is_local() => {
                // This is a call to a local function
                //
                // If there are free variables, we need to construct a fun and use that
                // as the callee; if there are no free variables, we can ignore the extra
                // transformation
                let mut callee = call.callee.as_local().unwrap();
                let callee_span = call.callee.span();
                let mut free = self.get_free(callee.function, callee.arity as usize);
                if !free.is_empty() {
                    // Build bif invocation that creates the fun with the closure environment
                    callee.arity += 1;
                    let op = FunctionName::new(symbols::Erlang, symbols::MakeFun, 2);
                    let mut mf_args = Vec::with_capacity(free.len() + 1);
                    mf_args.push(Expr::Local(Span::new(callee_span, callee)));
                    mf_args.append(&mut free);

                    let fun = Expr::Bif(Bif {
                        span: callee_span,
                        annotations: Annotations::default_compiler_generated(),
                        op,
                        args: mf_args,
                        ret: vec![],
                    });
                    let used = lit_list_vars(call.args.as_slice());
                    match brk {
                        Brk::Break(rs) => {
                            let _ = mem::replace(call.callee.as_mut(), fun);
                            call.ret = rs;
                            Ok((Expr::Call(call), used))
                        }
                        Brk::Return => {
                            let _ = mem::replace(call.callee.as_mut(), fun);
                            let enter = Expr::Enter(Enter {
                                span: call.span,
                                annotations: call.annotations,
                                callee: call.callee,
                                args: call.args,
                            });
                            Ok((enter, used))
                        }
                    }
                } else {
                    let used = lit_list_vars(call.args.as_slice());
                    match brk {
                        Brk::Break(rs) => {
                            call.ret = rs;
                            Ok((Expr::Call(call), used))
                        }
                        Brk::Return => {
                            let enter = Expr::Enter(Enter {
                                span: call.span,
                                annotations: call.annotations,
                                callee: call.callee,
                                args: call.args,
                            });
                            Ok((enter, used))
                        }
                    }
                }
            }
            Expr::Call(mut call) if brk.is_break() => {
                let used = sets::union(
                    op_vars(call.callee.as_ref()),
                    lit_list_vars(call.args.as_slice()),
                );
                call.ret = brk.into_ret();
                Ok((Expr::Call(call), used))
            }
            Expr::Call(Call {
                span,
                annotations,
                callee,
                args,
                ..
            }) => {
                let used = sets::union(op_vars(callee.as_ref()), lit_list_vars(args.as_slice()));
                let enter = Expr::Enter(Enter {
                    span,
                    annotations,
                    callee,
                    args,
                });
                Ok((enter, used))
            }
            Expr::Bif(mut bif) if brk.is_break() => {
                let used = lit_list_vars(bif.args.as_slice());
                let brs = self.bif_returns(bif.span, bif.op, brk.into_ret());
                bif.ret = brs;
                Ok((Expr::Bif(bif), used))
            }
            Expr::Match(Match {
                span,
                annotations,
                box body,
                ..
            }) => {
                let ret = brk.ret().to_vec();
                let (body, bu) = self.umatch(body, brk)?;
                Ok((
                    Expr::Match(Match {
                        span,
                        annotations,
                        body: Box::new(body),
                        ret,
                    }),
                    bu,
                ))
            }
            Expr::Try(Try {
                span,
                annotations,
                box arg,
                vars,
                box body,
                evars,
                box handler,
                ..
            }) if brk.is_break() => {
                let is_simple = match (vars.as_slice(), &body, &handler, brk.ret().is_empty()) {
                    ([v1], Expr::Var(v2), Expr::Literal(_), true) => v1 == v2,
                    _ => false,
                };
                if is_simple {
                    // This is a simple try/catch whose return value is
                    // ignored:
                    //
                    //   try E of V -> V when _:_:_ -> ignored_literal end, ...
                    //
                    // This is most probably a try/catch in a guard. To
                    // correctly handle the #k_test{} that ends the body of
                    // the guard, we MUST pass an empty list of break
                    // variables when processing the body.
                    let (a1, bu) = self.ubody(arg, Brk::Break(vec![]))?;
                    Ok((
                        Expr::Try(Try {
                            span,
                            annotations,
                            arg: Box::new(a1),
                            vars: vec![],
                            body: Box::new(Expr::Break(kbreak!(span))),
                            evars: vec![],
                            handler: Box::new(Expr::Break(kbreak!(span))),
                            ret: brk.into_ret(),
                        }),
                        bu,
                    ))
                } else {
                    // The general try/catch (in a guard or in body).
                    let mut avs = self.context.n_vars(vars.len(), Some(span));
                    let (arg, au) =
                        self.ubody(arg, Brk::Break(avs.drain(..).map(Expr::Var).collect()))?;
                    let (body, bu) = self.ubody(body, brk.clone())?;
                    let (handler, hu) = self.ubody(handler, brk.clone())?;
                    let var_set = vars.iter().map(|v| v.name).collect();
                    let evar_set = evars.iter().map(|v| v.name).collect();
                    let used = sets::union(
                        au,
                        sets::union(sets::subtract(bu, var_set), sets::subtract(hu, evar_set)),
                    );
                    Ok((
                        Expr::Try(Try {
                            span,
                            annotations,
                            arg: Box::new(arg),
                            vars,
                            body: Box::new(body),
                            evars,
                            handler: Box::new(handler),
                            ret: brk.into_ret(),
                        }),
                        used,
                    ))
                }
            }
            Expr::Try(Try {
                span,
                annotations,
                box arg,
                vars,
                box body,
                evars,
                box handler,
                ..
            }) if brk.is_return() => {
                let mut avs = self.context.n_vars(vars.len(), Some(span)); // Need dummy names here
                let (arg, au) =
                    self.ubody(arg, Brk::Break(avs.drain(..).map(Expr::Var).collect()))?; // Must break to clean up here
                let (body, bu) = self.ubody(body, Brk::Return)?;
                let (handler, hu) = self.ubody(handler, Brk::Return)?;
                let var_set = vars.iter().map(|v| v.name).collect();
                let evar_set = evars.iter().map(|v| v.name).collect();
                let used = sets::union(
                    au,
                    sets::union(sets::subtract(bu, var_set), sets::subtract(hu, evar_set)),
                );
                Ok((
                    Expr::TryEnter(TryEnter {
                        span,
                        annotations,
                        arg: Box::new(arg),
                        vars,
                        body: Box::new(body),
                        evars,
                        handler: Box::new(handler),
                    }),
                    used,
                ))
            }
            Expr::Catch(Catch {
                span,
                annotations,
                box body,
                ..
            }) if brk.is_break() => {
                let rb = self.context.next_var(Some(span));
                let (b1, bu) = self.ubody(body, Brk::Break(vec![Expr::Var(rb)]))?;
                // Guarantee _1_ return variable
                let mut ns = self.context.n_vars(1 - brk.ret().len(), Some(span));
                let mut ret = brk.into_ret();
                ret.extend(ns.drain(..).map(Expr::Var));
                Ok((
                    Expr::Catch(Catch {
                        span,
                        annotations,
                        body: Box::new(b1),
                        ret,
                    }),
                    bu,
                ))
            }
            Expr::Fun(IFun {
                span,
                annotations,
                mut vars,
                box body,
            }) if brk.is_break() => {
                // We do a transformation here which corresponds to the one in iletrec_funs.
                // We must also ensure that calls to closures append the closure value as an
                // extra argument, but that is done elsewhere.
                let (b1, bu) = self.ubody(body, Brk::Return)?; // Return out of new function
                let ns = vars.iter().map(|v| v.name).collect();
                let free = sets::subtract(bu, ns); // Free variables in fun
                let free_vars = free.iter().copied().map(Var::new).collect::<Vec<_>>();
                let mut fvs = free_vars.iter().cloned().map(Expr::Var).collect::<Vec<_>>();
                let closure_var = if free.is_empty() {
                    None
                } else {
                    let closure_var = self.context.next_var(Some(span));
                    vars.push(closure_var.clone());
                    Some(closure_var)
                };
                let arity = vars.len();
                let fname = match annotations.get(symbols::Id) {
                    Some(Annotation::Term(Literal {
                        value: Lit::Tuple(es),
                        ..
                    })) => {
                        // {_, _, name}
                        es[2].as_atom().unwrap()
                    }
                    _ => {
                        // No ide annotation, generate a name
                        self.context.new_fun_name(None)
                    }
                };
                // Create function definition
                let fname = FunctionName::new_local(fname, arity as u8);
                let body = if free.is_empty() {
                    b1
                } else {
                    self.unpack_closure_body(span, b1, closure_var.unwrap(), free_vars)
                };
                let function_annotations = if free.is_empty() {
                    annotations.clone()
                } else {
                    annotations.insert(symbols::Closure, Annotation::Unit)
                };
                let function = make_function(span, function_annotations, fname, vars, body);
                self.add_local_function(function);
                // Build bif invocation that creates the fun with the closure environment
                let op = FunctionName::new(symbols::Erlang, symbols::MakeFun, 2);
                let mut args = Vec::with_capacity(fvs.len() + 1);
                args.push(Expr::Local(Span::new(span, fname)));
                args.append(&mut fvs);

                // We know the value produced by this BIF is a function
                let mut ret = brk.into_ret();
                if !ret.is_empty() {
                    ret[0].set_type(Type::Term(TermType::Fun(None)));
                    if !free.is_empty() {
                        ret[0].annotations_mut().set(symbols::Closure);
                    }
                }

                Ok((
                    Expr::Bif(Bif {
                        span,
                        annotations,
                        op,
                        args,
                        ret,
                    }),
                    free,
                ))
            }
            Expr::Local(name) if brk.is_break() => {
                let span = name.span();
                let mut arity = name.arity as usize;
                let free = self.get_free(name.function, arity);
                let free = lit_list_vars(free.as_slice());
                let fvs = free
                    .iter()
                    .copied()
                    .map(|id| Expr::Var(Var::new(id)))
                    .collect::<Vec<_>>();
                let num_free = fvs.len();
                if num_free > 0 {
                    arity += 1;
                }
                let op = FunctionName::new(symbols::Erlang, symbols::MakeFun, 2);
                let mut args = Vec::with_capacity(num_free + 1);
                args.push(Expr::Local(Span::new(
                    span,
                    FunctionName::new_local(name.function, arity as u8),
                )));
                args.extend(fvs.iter().cloned());

                // We know the value produced by this BIF is a function
                let mut ret = brk.into_ret();
                if !ret.is_empty() {
                    ret[0].set_type(Type::Term(TermType::Fun(None)));
                }

                Ok((
                    Expr::Bif(Bif {
                        span,
                        annotations: Annotations::default(),
                        op,
                        args,
                        ret,
                    }),
                    free,
                ))
            }
            Expr::LetRecGoto(LetRecGoto {
                span,
                annotations,
                label,
                vars,
                box first,
                box then,
                ..
            }) => {
                let ret = brk.ret().to_vec();
                let ns = vars.iter().map(|v| v.name).collect();
                let (f1, fu) = self.ubody(first, brk.clone())?;
                let (t1, tu) = self.ubody(then, brk)?;
                let used = sets::subtract(sets::union(fu, tu), ns);
                Ok((
                    Expr::LetRecGoto(LetRecGoto {
                        span,
                        annotations,
                        label,
                        vars,
                        first: Box::new(f1),
                        then: Box::new(t1),
                        ret,
                    }),
                    used,
                ))
            }
            lit if brk.is_break() => {
                let span = lit.span();
                let annotations = lit.annotations().clone();
                // Transform literals to puts here.
                let used = lit_vars(&lit);
                let ret = self.ensure_return_vars(brk.into_ret());
                Ok((
                    Expr::Put(Put {
                        span,
                        annotations,
                        arg: Box::new(lit),
                        ret,
                    }),
                    used,
                ))
            }
            other => unimplemented!(
                "unexpected expression type in uexpr: {:#?} for {:#?}",
                &other,
                &brk
            ),
        }
    }

    /// Return all used variables for the body sequence.
    /// More efficient than `ubody` if it contains nested letrecs
    fn ubody_used_vars(&mut self, body: Expr) -> Result<RedBlackTreeSet<Ident>, ExprError> {
        // We need to save the current state which should be restored when returning
        // to the caller, since the caller just wants the used variables, not the side
        // effects on the current context
        let context = self.context.clone();
        self.context.ignore_funs = true;
        let result = self.ubody(body, Brk::Return);
        self.context = context;
        result.map(|(_, used)| used)
    }

    /// umatch(Match, Break, State) -> {Match,[UsedVar],State}.
    ///  Calculate the used variables for a match expression.
    fn umatch(
        &mut self,
        expr: Expr,
        brk: Brk,
    ) -> Result<(Expr, RedBlackTreeSet<Ident>), ExprError> {
        match expr {
            Expr::Alt(Alt {
                span,
                annotations,
                box first,
                box then,
            }) => {
                let (first, fu) = self.umatch(first, brk.clone())?;
                let (then, tu) = self.umatch(then, brk)?;
                let used = sets::union(fu, tu);
                Ok((
                    Expr::Alt(Alt {
                        span,
                        annotations,
                        first: Box::new(first),
                        then: Box::new(then),
                    }),
                    used,
                ))
            }
            Expr::Select(Select {
                span,
                annotations,
                var,
                types,
            }) => {
                let (types, tu) = self.umatch_type_clauses(types, brk)?;
                let used = tu.insert(var.name);
                Ok((
                    Expr::Select(Select {
                        span,
                        annotations,
                        var,
                        types,
                    }),
                    used,
                ))
            }
            Expr::Guard(Guard {
                span,
                annotations,
                clauses,
            }) => {
                let (clauses, used) = self.umatch_guard_clauses(clauses, brk)?;
                Ok((
                    Expr::Guard(Guard {
                        span,
                        annotations,
                        clauses,
                    }),
                    used,
                ))
            }
            pattern => self.ubody(pattern, brk),
        }
    }

    fn umatch_type_clause(
        &mut self,
        mut clause: TypeClause,
        brk: Brk,
    ) -> Result<(TypeClause, RedBlackTreeSet<Ident>), ExprError> {
        let clauses = clause.values.split_off(0);
        let (mut values, vu) = self.umatch_value_clauses(clauses, brk)?;
        clause.values.append(&mut values);
        Ok((clause, vu))
    }

    fn umatch_value_clause(
        &mut self,
        clause: ValueClause,
        brk: Brk,
    ) -> Result<(ValueClause, RedBlackTreeSet<Ident>), ExprError> {
        let (used, ps) = pat_vars(clause.value.as_ref());
        let (body, bu) = self.umatch(*clause.body, brk)?;
        let mut value = *clause.value;
        pat_anno_unused(&mut value, bu.clone(), ps.clone());
        let used = sets::union(used, sets::subtract(bu, ps));
        Ok((
            ValueClause {
                span: clause.span,
                annotations: clause.annotations,
                value: Box::new(value),
                body: Box::new(body),
            },
            used,
        ))
    }

    fn umatch_guard_clause(
        &mut self,
        clause: GuardClause,
        brk: Brk,
    ) -> Result<(GuardClause, RedBlackTreeSet<Ident>), ExprError> {
        let (guard, gu) = self.uexpr(*clause.guard, Brk::Break(vec![]))?;
        let (body, bu) = self.umatch(*clause.body, brk)?;
        let used = sets::union(gu, bu);
        Ok((
            GuardClause {
                span: clause.span,
                annotations: clause.annotations,
                guard: Box::new(guard),
                body: Box::new(body),
            },
            used,
        ))
    }

    fn umatch_type_clauses(
        &mut self,
        mut clauses: Vec<TypeClause>,
        brk: Brk,
    ) -> Result<(Vec<TypeClause>, RedBlackTreeSet<Ident>), ExprError> {
        let result = clauses
            .drain(..)
            .try_fold((vec![], rbt_set![]), |(mut ms, used), m| {
                let (m, mu) = self.umatch_type_clause(m, brk.clone())?;
                ms.push(m);
                let used = sets::union(mu, used);
                Ok((ms, used))
            })?;
        Ok(result)
    }

    fn umatch_value_clauses(
        &mut self,
        mut clauses: Vec<ValueClause>,
        brk: Brk,
    ) -> Result<(Vec<ValueClause>, RedBlackTreeSet<Ident>), ExprError> {
        let result = clauses
            .drain(..)
            .try_fold((vec![], rbt_set![]), |(mut ms, used), m| {
                let (m, mu) = self.umatch_value_clause(m, brk.clone())?;
                ms.push(m);
                let used = sets::union(mu, used);
                Ok((ms, used))
            })?;
        Ok(result)
    }

    fn umatch_guard_clauses(
        &mut self,
        mut clauses: Vec<GuardClause>,
        brk: Brk,
    ) -> Result<(Vec<GuardClause>, RedBlackTreeSet<Ident>), ExprError> {
        let result = clauses
            .drain(..)
            .try_fold((vec![], rbt_set![]), |(mut ms, used), m| {
                let (m, mu) = self.umatch_guard_clause(m, brk.clone())?;
                ms.push(m);
                let used = sets::union(mu, used);
                Ok((ms, used))
            })?;
        Ok(result)
    }

    fn add_local_function(&mut self, function: Function) {
        if self.context.ignore_funs {
            return;
        }
        if self.context.funs.iter().any(|f| f.name == function.name) {
            return;
        }
        self.context.funs.push(function);
    }

    /// get_free(Name, Arity, State) -> [Free].
    fn get_free(&mut self, name: Symbol, arity: usize) -> Vec<Expr> {
        let key = Name::Fun(name, arity);
        match self.context.free.get(&key) {
            None => vec![],
            Some(val) => val.clone(),
        }
    }

    /// store_free(Name, Arity, [Free], State) -> State.
    fn store_free(&mut self, name: Symbol, arity: usize, free: Vec<Expr>) {
        let key = Name::Fun(name, arity);
        self.context.free.insert_mut(key, free);
    }

    /// ensure_return_vars([Ret], State) -> {[Ret],State}.
    fn ensure_return_vars(&mut self, rets: Vec<Expr>) -> Vec<Expr> {
        if rets.is_empty() {
            vec![Expr::Var(self.context.next_var(None))]
        } else {
            rets
        }
    }

    fn bif_returns(
        &mut self,
        span: SourceSpan,
        callee: FunctionName,
        mut ret: Vec<Expr>,
    ) -> Vec<Expr> {
        assert!(
            callee.is_bif(),
            "expected bif callee to be a known bif, got {}",
            &callee
        );

        if callee.function == symbols::MatchFail {
            // This is used for effect only, and may have any number of returns
            return ret;
        }

        let sig = bifs::get(&callee).unwrap();
        // If this function raises, there are no values to return, but we need
        // to pretend there is one anyway if needed
        if sig.raises() {
            return ret;
        }

        let num_values = sig.results().len();
        let mut ns = self.context.n_vars(num_values - ret.len(), Some(span));
        ret.extend(ns.drain(..).map(Expr::Var));
        ret
    }
}

/// Make a Function, making sure that the body is always a Match.
fn make_function(
    span: SourceSpan,
    annotations: Annotations,
    name: FunctionName,
    vars: Vec<Var>,
    body: Expr,
) -> Function {
    match body {
        body @ Expr::Match(_) => Function {
            span,
            annotations,
            name,
            vars,
            body: Box::new(body),
        },
        body => {
            let anno = body.annotations().clone();
            let body = Box::new(Expr::Match(Match {
                span,
                annotations: anno,
                body: Box::new(body),
                ret: vec![],
            }));
            Function {
                span,
                annotations,
                name,
                vars,
                body,
            }
        }
    }
}

fn pat_anno_unused(pattern: &mut Expr, used: RedBlackTreeSet<Ident>, ps: RedBlackTreeSet<Ident>) {
    match pattern {
        Expr::Tuple(Tuple {
            ref mut elements, ..
        }) => {
            // Not extracting unused tuple elements is an optimization for
            // compile time and memory use during compilation. It is probably
            // worthwhile because it is common to extract only a few elements
            // from a huge record.
            let used = sets::intersection(used, ps);
            for element in elements.iter_mut() {
                match element {
                    Expr::Var(ref mut var) if !used.contains(&var.name) => {
                        var.annotations_mut().set(symbols::Unused);
                    }
                    _ => continue,
                }
            }
        }
        _ => (),
    }
}

/// op_vars(Op) -> [VarName].
fn op_vars(expr: &Expr) -> RedBlackTreeSet<Ident> {
    match expr {
        Expr::Remote(Remote::Dynamic(box Expr::Var(m), box Expr::Var(f))) => {
            rbt_set![m.name, f.name]
        }
        Expr::Remote(Remote::Dynamic(box Expr::Var(m), _)) => {
            rbt_set![m.name]
        }
        Expr::Remote(Remote::Dynamic(_, box Expr::Var(f))) => {
            rbt_set![f.name]
        }
        Expr::Remote(_) => rbt_set![],
        other => lit_vars(other),
    }
}

fn lit_list_vars(patterns: &[Expr]) -> RedBlackTreeSet<Ident> {
    patterns
        .iter()
        .fold(rbt_set![], |vars, p| sets::union(lit_vars(p), vars))
}

/// lit_vars(Literal) -> [VarName].
///  Return the variables in a literal.
fn lit_vars(expr: &Expr) -> RedBlackTreeSet<Ident> {
    match expr {
        Expr::Var(v) => rbt_set![v.name],
        Expr::Cons(Cons { head, tail, .. }) => {
            sets::union(lit_vars(head.as_ref()), lit_vars(tail.as_ref()))
        }
        Expr::Tuple(Tuple { elements, .. }) => lit_list_vars(elements.as_slice()),
        Expr::Map(Map { var, pairs, .. }) => {
            let set = lit_vars(var.as_ref());
            pairs.iter().fold(set, |vars, pair| {
                let ku = lit_vars(pair.key.as_ref());
                let vu = lit_vars(pair.value.as_ref());
                sets::union(sets::union(ku, vu), vars)
            })
        }
        Expr::Binary(Binary { segment, .. }) => lit_vars(segment.as_ref()),
        Expr::BinaryEnd(_) => rbt_set![],
        Expr::BinarySegment(BinarySegment {
            size: Some(sz),
            value,
            next,
            ..
        }) => sets::union(
            lit_vars(sz.as_ref()),
            sets::union(lit_vars(value.as_ref()), lit_vars(next.as_ref())),
        ),
        Expr::BinarySegment(BinarySegment { value, next, .. }) => {
            sets::union(lit_vars(value.as_ref()), lit_vars(next.as_ref()))
        }
        Expr::Literal(_) | Expr::Local(_) | Expr::Remote(Remote::Static(_)) => rbt_set![],
        Expr::Remote(Remote::Dynamic(m, f)) => {
            sets::union(lit_vars(m.as_ref()), lit_vars(f.as_ref()))
        }
        other => panic!("expected literal pattern, got {:?}", &other),
    }
}

/// pat_vars(Pattern) -> {[UsedVarName],[NewVarName]}.
///  Return variables in a pattern.  All variables are new variables
///  except those in the size field of binary segments and the key
///  field in map_pairs.
fn pat_vars(pattern: &Expr) -> (RedBlackTreeSet<Ident>, RedBlackTreeSet<Ident>) {
    match pattern {
        Expr::Var(v) => (rbt_set![], rbt_set![v.name]),
        Expr::Literal(_) => (rbt_set![], rbt_set![]),
        Expr::Cons(Cons { head, tail, .. }) => {
            let (used0, new0) = pat_vars(head.as_ref());
            let (used1, new1) = pat_vars(tail.as_ref());
            (sets::union(used0, used1), sets::union(new0, new1))
        }
        Expr::Tuple(Tuple { elements, .. }) => pat_list_vars(elements.as_slice()),
        Expr::Binary(Binary { segment, .. }) => pat_vars(segment.as_ref()),
        Expr::BinarySegment(BinarySegment {
            size: Some(sz),
            value,
            next,
            ..
        }) => {
            let (used0, new0) = pat_vars(value.as_ref());
            let (used1, new1) = pat_vars(next.as_ref());
            let (used, new) = (sets::union(used0, used1), sets::union(new0, new1));
            let (_, used2) = pat_vars(sz.as_ref());
            (sets::union(used, used2), new)
        }
        Expr::BinarySegment(BinarySegment { value, next, .. }) => {
            let (used0, new0) = pat_vars(value.as_ref());
            let (used1, new1) = pat_vars(next.as_ref());
            (sets::union(used0, used1), sets::union(new0, new1))
        }
        Expr::BinaryInt(BinarySegment {
            size: Some(sz),
            next,
            ..
        }) => {
            let (_, new) = pat_vars(next.as_ref());
            let (_, used) = pat_vars(sz.as_ref());
            (used, new)
        }
        Expr::BinaryInt(BinarySegment { next, .. }) => {
            let (_, new) = pat_vars(next.as_ref());
            (rbt_set![], new)
        }
        Expr::BinaryEnd(_) => (rbt_set![], rbt_set![]),
        Expr::Map(Map { pairs, .. }) => {
            pairs
                .iter()
                .fold((rbt_set![], rbt_set![]), |(used, new), pair| {
                    let (used1, new1) = pat_vars(pair.value.as_ref());
                    let (_, used2) = pat_vars(pair.key.as_ref());
                    (
                        sets::union(sets::union(used1, used2), used),
                        sets::union(new1, new),
                    )
                })
        }
        other => panic!("expected valid pattern expression, got {:?}", &other),
    }
}

fn pat_list_vars(patterns: &[Expr]) -> (RedBlackTreeSet<Ident>, RedBlackTreeSet<Ident>) {
    patterns
        .iter()
        .fold((rbt_set![], rbt_set![]), |(used, new), pattern| {
            let (used1, new1) = pat_vars(pattern);
            (sets::union(used, used1), sets::union(new, new1))
        })
}

#[allow(dead_code)]
fn integers(n: usize, m: usize) -> RangeInclusive<usize> {
    if n > m {
        // This produces an empty iterator
        return 1..=0;
    }
    n..=m
}
