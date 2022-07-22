use anyhow::bail;
use liblumen_number::Integer;
use liblumen_syntax_core::*;

use crate::ast;
use crate::evaluator;

use super::*;

impl<'m> LowerFunctionToCore<'m> {
    pub(super) fn lower_pattern<'a>(
        &mut self,
        builder: &'a mut IrBuilder,
        pattern: ast::Expr,
        input: Value,
        pattern_fail: Block,
        allow_unbound: bool,
    ) -> anyhow::Result<Value> {
        // When the failure path is a jump to the next function clause, then we don't need to pass any
        // block parameters. Clause entries never have block parameters, and standard pattern failure blocks
        // always have at least one, so we can use this to distinguish them.
        match pattern {
            ast::Expr::Var(var) => {
                let span = var.span();
                // If this is a wildcard, simply return the input
                if var.is_wildcard() {
                    return Ok(input);
                }
                // This is either a new binding or an alias pattern, check the current scope to confirm
                let sym = var.sym();
                if let Some(value) = builder.get_var(sym) {
                    // This is an alias pattern, so create an equality comparison in the current block with a
                    // conditional branch to the pattern failure block
                    let cond = builder.ins().eq_exact(input, value, span);
                    builder.ins().br_unless(cond, pattern_fail, &[input], span);
                } else {
                    // This is a new binding, so define the variable in the current scope
                    builder.define_var(sym, input);
                }
                // Return the input value as the output of this pattern
                Ok(input)
            }
            ast::Expr::Literal(lit) => {
                let pattern_span = lit.span();
                // A literal pattern simply requires us to lower the literal as an immediate/constant and then
                // compare to the input value for equality.
                let cond = match lit {
                    ast::Literal::Nil(_) => {
                        builder
                            .ins()
                            .eq_exact_imm(input, Immediate::Nil, pattern_span)
                    }
                    ast::Literal::Atom(l) => {
                        builder
                            .ins()
                            .eq_exact_imm(input, Immediate::Atom(l.name), pattern_span)
                    }
                    ast::Literal::String(l) => {
                        let is_list = builder.ins().is_type(
                            Type::Term(TermType::List(None)),
                            input,
                            pattern_span,
                        );
                        builder
                            .ins()
                            .br_unless(is_list, pattern_fail, &[input], pattern_span);
                        let s = l.as_str().get();
                        // TODO: This can probably be optimized in the future to avoid
                        // allocating a new list, and instead destructure and check each
                        // element bit by bit
                        let nil = builder.ins().nil(pattern_span);
                        let charlist = s.chars().rfold(nil, |tail, c| {
                            let head = builder.ins().character(c, pattern_span);
                            builder.ins().cons(head, tail, pattern_span)
                        });
                        builder.ins().eq_exact(input, charlist, pattern_span)
                    }
                    ast::Literal::Char(_, c) => {
                        builder.ins().eq_exact_imm(input, c.into(), pattern_span)
                    }
                    ast::Literal::Integer(_, Integer::Small(i)) => {
                        builder
                            .ins()
                            .eq_exact_imm(input, Immediate::Integer(i), pattern_span)
                    }
                    ast::Literal::Integer(_, big @ Integer::Big(_)) => {
                        let const_big = builder.make_constant(ConstantItem::Integer(big));
                        builder.ins().eq_exact_const(input, const_big, pattern_span)
                    }
                    ast::Literal::Float(_, f) => {
                        builder.ins().eq_exact_imm(input, f.into(), pattern_span)
                    }
                    ast::Literal::Cons(_, _h, _t) => {
                        // Allocate constant list, compare for equality
                        todo!()
                    }
                    ast::Literal::Tuple(_, _elements) => {
                        // Check if input is a tuple of arity N, bail early if not
                        // Allocate constant tuple, compare for equality
                        todo!()
                    }
                    ast::Literal::Map(_, _map) => {
                        // Check if input is a map, bail early if not
                        // For each key/value pair, check if the input map contains the pair
                        todo!()
                    }
                    ast::Literal::Binary(_, _bin) => {
                        // Allocate constant binary, compare for equality
                        todo!()
                    }
                };
                builder
                    .ins()
                    .br_unless(cond, pattern_fail, &[input], pattern_span);
                // Since this was effectively a guard on the input, use the input value as the output of this pattern
                Ok(input)
            }
            ast::Expr::Cons(cons) => {
                let span = cons.span;
                // This gets lowered as the following sequence:
                //
                // if !is_type(Input, cons), then pattern_fail
                // if Input.head != Pattern.head, then pattern_fail
                // if Input.tail != Pattern.tail, then pattern_fail
                let is_type = builder
                    .ins()
                    .is_type(Type::Term(TermType::List(None)), input, span);
                builder
                    .ins()
                    .br_unless(is_type, pattern_fail, &[input], span);
                let list = builder
                    .ins()
                    .cast(input, Type::Term(TermType::List(None)), span);
                // Match the head
                let hd = builder.ins().head(list, span);
                self.lower_pattern(builder, *cons.head, hd, pattern_fail, allow_unbound)?;
                // Match the tail
                let tl = builder.ins().tail(list, span);
                self.lower_pattern(builder, *cons.tail, tl, pattern_fail, allow_unbound)?;
                Ok(list)
            }
            ast::Expr::Tuple(tuple) => {
                let span = tuple.span;
                // This gets lowered as the following sequence:
                //
                // if !is_type(Input, tuple(Arity)), then pattern_fail
                // E0 = get_element(Input, 1)
                // if E0 != Pattern[1], then pattern_fail
                // ..
                let tuple_type = Type::tuple(tuple.elements.len());
                let is_type = builder.ins().is_type(tuple_type.clone(), input, span);
                builder
                    .ins()
                    .br_unless(is_type, pattern_fail, &[input], span);
                let input = builder.ins().cast(input, tuple_type, span);
                // Check each element
                for (index, element) in tuple.elements.iter().enumerate() {
                    let element_value = builder.ins().get_element_imm(
                        input,
                        Immediate::Integer(index.try_into().unwrap()),
                        element.span(),
                    );
                    self.lower_pattern(
                        builder,
                        element.clone(),
                        element_value,
                        pattern_fail,
                        allow_unbound,
                    )?;
                }
                Ok(input)
            }
            ast::Expr::Map(mut expr) => {
                let span = expr.span;
                let is_type = builder
                    .ins()
                    .is_type(Type::Term(TermType::Map), input, span);
                builder
                    .ins()
                    .br_unless(is_type, pattern_fail, &[input], span);
                let map = builder.ins().cast(input, Type::Term(TermType::Map), span);
                for field in expr.fields.drain(..) {
                    match field {
                        ast::MapField::Exact { span, key, value } => {
                            // Get the key value, which should be a guard expression
                            let key = self.lower_expr(builder, key)?;
                            // Fetch the value of that key from the map, failing the pattern if the key is not present
                            let value_input = builder.ins().map_get(map, key, span);
                            let is_none =
                                builder
                                    .ins()
                                    .eq_exact_imm(value_input, Immediate::None, span);
                            builder.ins().br_if(is_none, pattern_fail, &[input], span);
                            // Lastly, lower the pattern match against the fetched value
                            self.lower_pattern(
                                builder,
                                value,
                                value_input,
                                pattern_fail,
                                allow_unbound,
                            )?;
                        }
                        _ => {
                            // Our parser does not permit Assoc keys in patterns
                            unreachable!()
                        }
                    }
                }
                Ok(map)
            }
            ast::Expr::Binary(_bin) => {
                todo!("support for binary matching is not implemented")
            }
            ast::Expr::BinaryExpr(binop) if binop.op == ast::BinaryOp::Append => {
                let span = binop.span;
                match *binop.lhs {
                    // String prefix match
                    ast::Expr::Literal(ast::Literal::String(s)) => {
                        // We need to lower this as a chain of successive cons patterns that
                        // match each character in the prefix, before finally evaluating the
                        // remaining tail value against the right-hand side pattern
                        let span = s.span;
                        let s = s.as_str().get();

                        let mut current = input;
                        let mut initial = None;
                        for c in s.chars() {
                            // Verify the input term is a list
                            let is_type = builder.ins().is_type(
                                Type::Term(TermType::List(None)),
                                current,
                                span,
                            );
                            builder
                                .ins()
                                .br_unless(is_type, pattern_fail, &[input], span);
                            let list =
                                builder
                                    .ins()
                                    .cast(input, Type::Term(TermType::List(None)), span);
                            // Make sure we save the result of casting the initial list input
                            if initial.is_none() {
                                initial = Some(list);
                            }
                            // Extract the head value and match it against the current character
                            let hd = builder.ins().head(list, span);
                            let cond =
                                builder
                                    .ins()
                                    .eq_exact_imm(hd, Immediate::Integer(c as i64), span);
                            builder.ins().br_unless(cond, pattern_fail, &[input], span);
                            // Extract the tail for the next iteration
                            current = builder.ins().tail(list, span);
                        }

                        let rhs = self.lower_pattern(
                            builder,
                            *binop.rhs,
                            current,
                            pattern_fail,
                            allow_unbound,
                        )?;

                        // If the initial content was non-empty, return the result of casting it to a list
                        // However, if the initial content was empty, then the result of the expression is
                        // given by the right-hand operand
                        Ok(initial.unwrap_or(rhs))
                    }
                    // [] ++ Expr = Input
                    ast::Expr::Literal(ast::Literal::Nil(_)) => {
                        // This pattern is equivalent to a pattern consisting of only the right-hand operand
                        return self.lower_pattern(
                            builder,
                            *binop.rhs,
                            input,
                            pattern_fail,
                            allow_unbound,
                        );
                    }
                    // [H | T] ++ Expr = Input
                    cons @ ast::Expr::Cons(_) => {
                        // This pattern is only valid if this is a proper list of literals
                        let span = cons.span();
                        let mut current = cons;
                        let mut rest = input;
                        let mut initial = None;
                        loop {
                            match current {
                                ast::Expr::Cons(ast::Cons { span, head, tail }) => {
                                    let is_type = builder.ins().is_type(
                                        Type::Term(TermType::List(None)),
                                        rest,
                                        span,
                                    );
                                    builder
                                        .ins()
                                        .br_unless(is_type, pattern_fail, &[input], span);
                                    let list = builder.ins().cast(
                                        input,
                                        Type::Term(TermType::List(None)),
                                        span,
                                    );
                                    // On the first iteration, current is always a list, and we want to
                                    // reuse the result of casting the input value as a list
                                    if initial.is_none() {
                                        initial = Some(list);
                                    }
                                    let hd = builder.ins().head(list, span);
                                    match *head {
                                        head @ ast::Expr::Literal(_)
                                        | head @ ast::Expr::Cons(_)
                                        | head @ ast::Expr::Tuple(_)
                                        | head @ ast::Expr::Map(_)
                                        | head @ ast::Expr::Binary(_) => {
                                            self.lower_pattern(
                                                builder,
                                                head,
                                                hd,
                                                pattern_fail,
                                                allow_unbound,
                                            )?;
                                        }
                                        other => {
                                            self.show_error(
                                                "illegal pattern",
                                                &[
                                                    (span, "in this pattern"),
                                                    (other.span(), "expected literal here"),
                                                ],
                                            );
                                            bail!("invalid expression in pattern context")
                                        }
                                    }
                                    current = *tail;
                                    rest = builder.ins().tail(list, span);
                                }
                                ast::Expr::Literal(ast::Literal::Nil(_)) => {
                                    break;
                                }
                                other => {
                                    self.show_error(
                                        "illegal pattern",
                                        &[
                                            (span, "in this pattern"),
                                            (other.span(), "expected nil or a list here"),
                                        ],
                                    );
                                    bail!("invalid expression in pattern context")
                                }
                            }
                        }
                        self.lower_pattern(builder, *binop.rhs, rest, pattern_fail, allow_unbound)?;
                        Ok(initial.unwrap())
                    }
                    other => {
                        self.show_error(
                            "invalid pattern expression",
                            &[
                                (span, "in this pattern"),
                                (other.span(), "invalid operand for ++ operator, expected literal string or list")
                            ]
                        );
                        bail!("invalid expression in pattern context")
                    }
                }
            }
            ast::Expr::BinaryExpr(ref binop) => {
                let span = binop.span;
                // Since only arithmetic/bitwise ops are valid here, the output term must be numeric
                let cond =
                    match evaluator::eval_expr(&pattern, None) {
                        Ok(ast::Literal::Integer(_, Integer::Small(i))) => builder
                            .ins()
                            .neq_exact_imm(input, Immediate::Integer(i), span),
                        Ok(ast::Literal::Integer(_, big @ Integer::Big(_))) => {
                            let const_big = builder.make_constant(ConstantItem::Integer(big));
                            builder.ins().neq_exact_const(input, const_big, span)
                        }
                        Ok(ast::Literal::Float(_, f)) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Float(f.inner()), span)
                        }
                        Ok(_) => panic!(
                        "unexpected non-numeric output when evaluating constant numeric expression"
                    ),
                        Err(e) => {
                            let eval_span = e.span();
                            let eval_message = e.to_string();
                            self.show_error(
                                "invalid constant expression in pattern context",
                                &[
                                    (span, "in this expression"),
                                    (eval_span, eval_message.as_str()),
                                ],
                            );
                            bail!("invalid expression in pattern context: {}", e)
                        }
                    };
                builder.ins().br_if(cond, pattern_fail, &[input], span);
                Ok(input)
            }
            ast::Expr::UnaryExpr(ast::UnaryExpr {
                span,
                op,
                ref operand,
            }) if op.is_valid_in_patterns() => {
                let cond =
                    match evaluator::eval_expr(operand, None) {
                        Ok(ast::Literal::Integer(_, Integer::Small(i))) => builder
                            .ins()
                            .neq_exact_imm(input, Immediate::Integer(i), span),
                        Ok(ast::Literal::Integer(_, big @ Integer::Big(_))) => {
                            let const_big = builder.make_constant(ConstantItem::Integer(big));
                            builder.ins().neq_exact_const(input, const_big, span)
                        }
                        Ok(ast::Literal::Float(_, f)) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Float(f.inner()), span)
                        }
                        Ok(_) => panic!(
                        "unexpected non-numeric output when evaluating constant numeric expression"
                    ),
                        Err(e) => {
                            let eval_span = e.span();
                            let eval_message = e.to_string();
                            self.show_error(
                                "invalid constant expression in pattern context",
                                &[
                                    (span, "in this expression"),
                                    (eval_span, eval_message.as_str()),
                                ],
                            );
                            bail!("invalid expression in pattern context: {}", e)
                        }
                    };
                builder.ins().br_if(cond, pattern_fail, &[input], span);
                Ok(input)
            }
            ast::Expr::Record(_) | ast::Expr::RecordAccess(_) => panic!(
                "record patterns should have been lowered to tuple ops by now: {:?}",
                &pattern
            ),
            invalid => bail!("invalid expression in pattern context: {:?}", &invalid),
        }
    }
}
