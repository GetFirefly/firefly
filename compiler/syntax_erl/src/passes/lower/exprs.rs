use std::collections::BTreeMap;

use anyhow::bail;
use either::Either;
use liblumen_intern::{symbols, Symbol};
use liblumen_number::Integer;
use liblumen_syntax_core::*;
use log::debug;

use crate::ast;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_block_expr<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: ast::Expr,
    ) -> anyhow::Result<(Value, BTreeMap<Symbol, Value>)> {
        let mut exports = BTreeMap::new();
        let value = match expr {
            ast::Expr::Match(expr) => self.lower_match_block(builder, expr, &mut exports)?,
            ast::Expr::Begin(expr) => {
                let (result, exported) = self.lower_block(builder, expr.body)?;
                for (binding, value) in exported.iter() {
                    debug_assert!(!exports.contains_key(binding), "begin block exported a binding for '{}' when it should have been treated as a match", binding);
                    exports.insert(*binding, *value);
                }
                result
            }
            ast::Expr::If(expr) => self.lower_if_block(builder, expr, &mut exports)?,
            ast::Expr::Case(expr) => self.lower_case_block(builder, expr, &mut exports)?,
            invalid => bail!("not a block expression: {:?}", &invalid),
        };
        Ok((value, exports))
    }

    pub(super) fn lower_expr<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: ast::Expr,
    ) -> anyhow::Result<Value> {
        match expr {
            ast::Expr::Var(var) => self.lower_var(builder, var),
            ast::Expr::Literal(lit) => self.lower_literal(builder, lit),
            ast::Expr::Nil(nil) => Ok(builder.ins().nil(nil.span())),
            ast::Expr::Cons(cons) => self.lower_cons(builder, cons),
            ast::Expr::Tuple(tuple) => self.lower_tuple(builder, tuple),
            ast::Expr::Map(map) => self.lower_map(builder, map),
            ast::Expr::MapUpdate(map) => self.lower_map_update(builder, map),
            ast::Expr::Binary(bin) => self.lower_binary(builder, bin),
            ast::Expr::ListComprehension(lc) => self.lower_lc(builder, lc),
            ast::Expr::BinaryComprehension(bc) => self.lower_bc(builder, bc),
            ast::Expr::Apply(apply) => self.lower_apply(builder, apply, false),
            ast::Expr::Match(expr) => self.lower_match(builder, expr),
            ast::Expr::Catch(expr) => self.lower_catch(builder, expr),
            ast::Expr::BinaryExpr(expr) => self.lower_binary_op(builder, expr),
            ast::Expr::UnaryExpr(expr) => self.lower_unary_op(builder, expr),
            ast::Expr::Try(expr) => self.lower_try(builder, expr),
            ast::Expr::Fun(fun) => self.lower_fun(builder, fun),
            ast::Expr::Receive(receive) => self.lower_receive(builder, receive),
            ast::Expr::If(_) | ast::Expr::Begin(_) | ast::Expr::Case(_) => {
                self.lower_block_expr(builder, expr).map(|(value, _)| value)
            }
            invalid => bail!("invalid expression in current context: {:?}", &invalid),
        }
    }

    /// Variable references simply change the value of `last_inst_result` to be
    /// the current value bound to `ident` in scope
    pub(super) fn lower_var<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        var: ast::Var,
    ) -> anyhow::Result<Value> {
        let span = var.span();
        let sym = var.sym();
        if let Some(v) = builder.get_var(sym) {
            return Ok(v);
        }

        self.show_error("undefined variable", &[(span, "used here")]);
        bail!("undefined variable '{}'", sym)
    }

    /// Literals are directly translated as constant expressions
    pub(super) fn lower_literal<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        lit: ast::Literal,
    ) -> anyhow::Result<Value> {
        match lit {
            ast::Literal::Atom(ident) => Ok(builder.ins().atom(ident.name, ident.span)),
            ast::Literal::String(ident) => {
                let span = ident.span;
                let value = ident.as_str().get();
                let nil = builder.ins().nil(span);
                let charlist = value.chars().rfold(nil, |tail, c| {
                    let head = builder.ins().character(c, span);
                    builder.ins().cons(head, tail, span)
                });
                Ok(charlist)
            }
            ast::Literal::Char(span, c) => Ok(builder.ins().character(c, span)),
            ast::Literal::Integer(span, Integer::Small(i)) => Ok(builder.ins().int(i, span)),
            ast::Literal::Integer(span, Integer::Big(i)) => Ok(builder.ins().bigint(i, span)),
            ast::Literal::Float(span, f) => Ok(builder.ins().float(f.inner(), span)),
        }
    }

    pub(super) fn lower_cons<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        cons: ast::Cons,
    ) -> anyhow::Result<Value> {
        // Since cons cells can be deeply recursive, we avoid potential stack overflow
        // by constructing a vector of all the items, and then once we've hit the last
        // item, folding the vector into a series of cons cells
        let mut items = Vec::with_capacity(2);
        items.push((cons.span, *cons.head));
        // Unroll the full set of nested cons cells
        let mut tail = *cons.tail;
        while let ast::Expr::Cons(cell) = tail {
            items.push((cell.span, *cell.head));
            tail = *cell.tail;
        }
        // Lower expressions from left-to-right, i.e. from the bottom of the list, up
        let tail = self.lower_expr(builder, tail)?;
        let result =
            items
                .drain(0..)
                .try_rfold::<_, _, anyhow::Result<Value>>(tail, |tl, (span, hd)| {
                    let hd = self.lower_expr(builder, hd)?;
                    Ok(builder.ins().cons(hd, tl, span))
                })?;

        Ok(result)
    }

    pub(super) fn lower_tuple<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut tuple: ast::Tuple,
    ) -> anyhow::Result<Value> {
        let span = tuple.span;

        // Allocate a tuple of the appropriate size
        let result = builder.ins().tuple_imm(tuple.elements.len(), span);
        // For each element, write the resulting value to the tuple
        for (i, element) in tuple.elements.drain(0..).enumerate() {
            let span = element.span();
            let value = self.lower_expr(builder, element)?;
            let index = builder.ins().int((i + 1).try_into().unwrap(), span);
            builder.ins().set_element(result, index, value, span);
        }

        Ok(result)
    }

    pub(super) fn lower_map<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut map: ast::Map,
    ) -> anyhow::Result<Value> {
        let span = map.span;

        // Allocate a map
        let result = builder.ins().map(span);
        // Mutate the map using the given field ops
        for field in map.fields.drain(0..) {
            match field {
                ast::MapField::Assoc { span, key, value } => {
                    let key = self.lower_expr(builder, key)?;
                    let value = self.lower_expr(builder, value)?;
                    builder.ins().map_put(result, key, value, span);
                }
                ast::MapField::Exact { span, key, value } => {
                    let key = self.lower_expr(builder, key)?;
                    let value = self.lower_expr(builder, value)?;
                    builder.ins().map_update(result, key, value, span);
                }
            }
        }

        Ok(result)
    }

    pub(super) fn lower_map_update<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        mut expr: ast::MapUpdate,
    ) -> anyhow::Result<Value> {
        let map = self.lower_expr(builder, *expr.map)?;

        for update in expr.updates.drain(0..) {
            match update {
                ast::MapField::Assoc { span, key, value } => {
                    let key = self.lower_expr(builder, key)?;
                    let value = self.lower_expr(builder, value)?;
                    builder.ins().map_put(map, key, value, span);
                }
                ast::MapField::Exact { span, key, value } => {
                    let key = self.lower_expr(builder, key)?;
                    let value = self.lower_expr(builder, value)?;
                    builder.ins().map_update(map, key, value, span);
                }
            }
        }

        Ok(map)
    }

    pub(super) fn lower_binary<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _bin: ast::Binary,
    ) -> anyhow::Result<Value> {
        todo!()
    }

    pub(super) fn lower_lc<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _expr: ast::ListComprehension,
    ) -> anyhow::Result<Value> {
        todo!()
    }

    pub(super) fn lower_bc<'a>(
        &mut self,
        _builder: &'a mut IrBuilder,
        _expr: ast::BinaryComprehension,
    ) -> anyhow::Result<Value> {
        todo!()
    }

    pub(super) fn lower_apply<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        apply: ast::Apply,
        is_tail: bool,
    ) -> anyhow::Result<Value> {
        let span = apply.span;
        let callee_span = apply.callee.span();
        let mut args = apply.args;
        let arity: u8 = args.len().try_into().unwrap();

        // Lower arguments
        let mut argv = Vec::with_capacity(args.len());
        for arg in args.drain(0..) {
            let value = self.lower_expr(builder, arg)?;
            argv.push(value);
        }

        // Lower the callee/call itself
        match *apply.callee {
            ast::Expr::Literal(ast::Literal::Atom(f)) => {
                let fun =
                    self.resolve_local(FunctionName::new_local(f.name, arity), callee_span)?;
                self.lower_static_apply(builder, fun, argv, callee_span, span)
            }
            ast::Expr::Var(ast::Var(v)) => {
                if let Some(fun) = builder.get_func_name(v.name) {
                    return self.lower_static_apply(builder, fun, argv, callee_span, span);
                }
                if let Some(value) = builder.get_var(v.name) {
                    self.lower_dynamic_apply(builder, None, value, argv, callee_span, span)
                } else {
                    let message = format!("undefined variable '{}'", v);
                    self.show_error(
                        message.as_str(),
                        &[
                            (v.span, "this is not bound in the current scope"),
                            (callee_span, "used here"),
                        ],
                    );
                    bail!("undefined variable '{}'", v)
                }
            }
            ast::Expr::FunctionName(ast::FunctionName::Resolved(fun)) => {
                self.lower_static_apply(builder, *fun, argv, callee_span, span)
            }
            ast::Expr::FunctionName(ast::FunctionName::PartiallyResolved(fun)) => {
                let message = format!("function {} is undefined", &fun);
                self.show_error_annotated(
                    message.as_str(),
                    &[(fun.span(), "found here")],
                    &["verify it is defined locally, or consider importing this function"],
                );
                bail!("invalid callee, no such function defined in scope")
            }
            ast::Expr::FunctionName(ast::FunctionName::Unresolved(unresolved)) => {
                let callee_span = unresolved.span;
                let module = match unresolved.module {
                    Some(m) => Some(self.resolve_name(builder, m)?),
                    None => None,
                };
                let function = self.resolve_name(builder, unresolved.function)?;
                let dynamic_arity = self.resolve_arity(builder, unresolved.arity)?;
                match (module, function, dynamic_arity) {
                    // fun foo:bar/1
                    (Some(Either::Left(m)), Either::Left(f), Either::Left(a)) => {
                        let fun = FunctionName::new(m.name, f.name, a);
                        self.lower_static_apply(builder, fun, argv, callee_span, span)
                    }
                    // fun bar/1
                    (None, Either::Left(f), Either::Left(a)) => {
                        let fun = self.resolve_local(FunctionName::new_local(f.name, a), callee_span)?;
                        self.lower_static_apply(builder, fun, argv, callee_span, span)
                    }
                    // fun M:bar/1
                    (Some(Either::Right(m)), Either::Left(f), Either::Left(a)) if a == arity => {
                        let function = builder.ins().atom(f.name, f.span);
                        self.lower_dynamic_apply(builder, Some(m), function, argv, callee_span, span)
                    }
                    (Some(Either::Right(m)), Either::Left(f), Either::Left(a)) => {
                        let expected_message = format!("this function has an arity of {}", a);
                        let given_message = format!("but was called with {}", arity);
                        self.show_error("incorrect number of arguments for callee", &[(callee_span, expected_message.as_str()), (span, given_message.as_str())]);
                        let function = builder.ins().atom(f.name, f.span);
                        self.lower_dynamic_apply(builder, Some(m), function, argv, callee_span, span)
                    }
                    // fun M:F/1
                    (Some(Either::Right(m)), Either::Right(f), Either::Left(a)) if a == arity => {
                        self.lower_dynamic_apply(builder, Some(m), f, argv, callee_span, span)
                    }
                    (Some(Either::Right(m)), Either::Right(f), Either::Left(a)) => {
                        let expected_message = format!("this function has an arity of {}", a);
                        let given_message = format!("but was called with {}", arity);
                        self.show_error("incorrect number of arguments for callee", &[(callee_span, expected_message.as_str()), (span, given_message.as_str())]);
                        self.lower_dynamic_apply(builder, Some(m), f, argv, callee_span, span)
                    }
                    // fun M:F/A
                    (Some(Either::Right(m)), Either::Right(f), _) => {
                        self.lower_dynamic_apply(builder, Some(m), f, argv, callee_span, span)
                    }
                    invalid => panic!("invalid syntax found during lowering, function name has invalid construction: {:?}", invalid),
                }
            }
            ast::Expr::Remote(remote) => {
                if let Ok(fun) = remote.try_eval(arity) {
                    debug!(
                        "successfully transformed remote to constant function reference: {}",
                        &fun
                    );
                    self.lower_static_apply(builder, fun, argv, callee_span, span)
                } else {
                    let module = self.lower_expr(builder, *remote.module)?;
                    let function = self.lower_expr(builder, *remote.function)?;
                    self.lower_dynamic_apply(
                        builder,
                        Some(module),
                        function,
                        argv,
                        callee_span,
                        span,
                    )
                }
            }
            invalid => {
                self.show_error(
                    "invalid callee, expected function name or remote",
                    &[(invalid.span(), "in this function application")],
                );
                bail!(
                    "invalid callee, expected function name or remote, got {:?}",
                    invalid
                )
            }
        }
    }

    pub(super) fn lower_match<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: ast::Match,
    ) -> anyhow::Result<Value> {
        let value = self.lower_expr(builder, *expr.expr)?;
        match *expr.pattern {
            ast::Expr::Var(var) => {
                let sym = var.sym();
                if let Some(_bound) = builder.get_var(sym) {
                    // This variable is bound in scope, so this is actually an
                    // equality assertion
                    todo!("pattern matching against previously bound vars with '='")
                } else {
                    // This variable is not bound in scope, so add a new binding
                    builder.define_var(sym, value);
                    Ok(value)
                }
            }
            _pattern => todo!("pattern matching against expressions with '='"),
        }
    }

    pub(super) fn lower_catch<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        expr: ast::Catch,
    ) -> anyhow::Result<Value> {
        let span = expr.span;
        // Catch requires us to split control flow such that the exceptional and non-exceptional
        // paths join back up on the return path. Exceptions should jump to a landing pad in which
        // the exception value is converted to a term
        let current_block = builder.current_block();
        // The landing pad is where control will land when an exception occurs
        // It receives a single block argument which is the exception to handle
        let landing_pad = builder.create_block();
        let exception = builder.append_block_param(landing_pad, Type::Term(TermType::Any), span);
        // The result block is where the fork in control is rejoined, it receives a single block argument which is
        // either the normal return value, or the caught/wrapped exception value
        let result_block = builder.create_block();
        let result = builder.append_block_param(result_block, Type::Term(TermType::Any), span);
        // The exit block handles wrapping exit/error reasons in the {'EXIT', Reason} tuple
        // It receives a single block argument which corresponds to `Reason` in the previous sentence.
        let exit_block = builder.create_block();
        let exit_reason = builder.append_block_param(exit_block, Type::Term(TermType::Any), span);

        // Now, in the landing pad, check what type of exception we have and branch accordingly
        builder.switch_to_block(landing_pad);
        let exception = builder.ins().cast(exception, Type::Exception, span);
        // All three types of exception require us to have the class and reason handy
        let class = builder.ins().exception_class(exception, span);
        let reason = builder.ins().exception_reason(exception, span);
        // Throws are the most common, and require no special handling, so we jump straight to the result block for them
        let is_throw = builder
            .ins()
            .eq_exact_imm(class, Immediate::Atom(symbols::Throw), span);
        builder.ins().br_if(is_throw, result_block, &[reason], span);
        // Exits are the next simplest, as we just wrap the reason in a tuple, so we jump straight to the exit block
        let is_exit = builder
            .ins()
            .eq_exact_imm(class, Immediate::Atom(symbols::Exit), span);
        builder.ins().br_if(is_exit, exit_block, &[reason], span);
        // Errors are handled in the landing pad directly
        let trace = builder.ins().exception_trace(exception, span);
        // We have to construct a new error reason, and then jump to the exit block to wrap it in the exit tuple
        let error_reason = builder.ins().tuple_imm(2, span);
        let one = builder.ins().int(1, span);
        let error_reason = builder.ins().set_element(error_reason, one, reason, span);
        let two = builder.ins().int(2, span);
        let error_reason = builder.ins().set_element(error_reason, two, trace, span);
        builder.ins().br(exit_block, &[error_reason], span);

        // In the exit block, we need just to construct the {'EXIT', Reason} tuple, and then jump to the result block
        builder.switch_to_block(exit_block);
        let wrapped_reason = builder.ins().tuple_imm(2, span);
        let wrapped_reason =
            builder
                .ins()
                .set_element_imm(wrapped_reason, 1, Immediate::Atom(symbols::EXIT), span);
        let two = builder.ins().int(2, span);
        let wrapped_reason = builder
            .ins()
            .set_element(wrapped_reason, two, exit_reason, span);
        builder.ins().br(result_block, &[wrapped_reason], span);

        // Lower the inner expression in the starting block, with the `catch` landing pad at the top of the stack
        builder.switch_to_block(current_block);
        self.landing_pads.push(landing_pad);
        let ret = self.lower_expr(builder, *expr.expr)?;

        // We're exiting the catch now, so pop the landing pad from the stack, and branch to the result block with
        // the normal return value
        self.landing_pads.pop();
        builder.ins().br(result_block, &[ret], span);

        // We leave off in the result block, using its block argument as the return value of the catch expression
        builder.switch_to_block(result_block);

        Ok(result)
    }

    pub(super) fn lower_binary_op<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        op: ast::BinaryExpr,
    ) -> anyhow::Result<Value> {
        // We attempt to always evaluate expressions in left-to-right order
        let span = op.span;
        let operator = op.op;
        let lhs = self.lower_expr(builder, *op.lhs)?;
        let rhs = self.lower_expr(builder, *op.rhs)?;
        let result = match operator {
            ast::BinaryOp::Send => {
                let send2 = FunctionName::new(symbols::Erlang, symbols::Send, 2);
                self.lower_static_apply(builder, send2, vec![lhs, rhs], span, span)?
            }
            ast::BinaryOp::Equal => builder.ins().eq(lhs, rhs, span),
            ast::BinaryOp::NotEqual => builder.ins().neq(lhs, rhs, span),
            ast::BinaryOp::StrictEqual => builder.ins().eq_exact(lhs, rhs, span),
            ast::BinaryOp::StrictNotEqual => builder.ins().neq_exact(lhs, rhs, span),
            ast::BinaryOp::Lte => builder.ins().lte(lhs, rhs, span),
            ast::BinaryOp::Lt => builder.ins().lt(lhs, rhs, span),
            ast::BinaryOp::Gte => builder.ins().gte(lhs, rhs, span),
            ast::BinaryOp::Gt => builder.ins().gt(lhs, rhs, span),
            ast::BinaryOp::Append => builder.ins().list_concat(lhs, rhs, span),
            ast::BinaryOp::Remove => builder.ins().list_subtract(lhs, rhs, span),
            ast::BinaryOp::Add => builder.ins().add(lhs, rhs, span),
            ast::BinaryOp::Sub => builder.ins().sub(lhs, rhs, span),
            ast::BinaryOp::Multiply => builder.ins().mul(lhs, rhs, span),
            ast::BinaryOp::Divide => builder.ins().fdiv(lhs, rhs, span),
            ast::BinaryOp::Div => builder.ins().div(lhs, rhs, span),
            ast::BinaryOp::Rem => builder.ins().rem(lhs, rhs, span),
            ast::BinaryOp::And => builder.ins().and(lhs, rhs, span),
            ast::BinaryOp::AndAlso => builder.ins().andalso(lhs, rhs, span),
            ast::BinaryOp::Or => builder.ins().or(lhs, rhs, span),
            ast::BinaryOp::OrElse => builder.ins().orelse(lhs, rhs, span),
            ast::BinaryOp::Xor => builder.ins().xor(lhs, rhs, span),
            ast::BinaryOp::Band => builder.ins().band(lhs, rhs, span),
            ast::BinaryOp::Bor => builder.ins().bor(lhs, rhs, span),
            ast::BinaryOp::Bxor => builder.ins().bxor(lhs, rhs, span),
            ast::BinaryOp::Bsl => builder.ins().bsl(lhs, rhs, span),
            ast::BinaryOp::Bsr => builder.ins().bsr(lhs, rhs, span),
        };
        Ok(result)
    }

    pub(super) fn lower_unary_op<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        op: ast::UnaryExpr,
    ) -> anyhow::Result<Value> {
        let span = op.span;
        let operator = op.op;
        let operand = self.lower_expr(builder, *op.operand)?;
        let result = match operator {
            ast::UnaryOp::Plus => operand,
            ast::UnaryOp::Minus => builder.ins().neg(operand, span),
            ast::UnaryOp::Bnot => builder.ins().bnot(operand, span),
            ast::UnaryOp::Not => builder.ins().not(operand, span),
        };
        Ok(result)
    }
}
