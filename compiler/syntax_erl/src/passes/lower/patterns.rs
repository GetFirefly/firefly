use anyhow::bail;
use liblumen_intern::symbols;
use liblumen_number::{Integer, Number};
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
    ) -> anyhow::Result<Value> {
        // When the failure path is a jump to the next function clause, then we don't need to pass any
        // block parameters. Clause entries never have block parameters, and standard pattern failure blocks
        // always have at least one, so we can use this to distinguish them.
        match pattern {
            ast::Expr::Var(var) => {
                let sym = var.sym();
                // If this is a wildcard, simply return the input
                if sym == symbols::WildcardMatch {
                    return Ok(input);
                }
                // This is either a new binding or an alias pattern, check the current scope to confirm
                if let Some(value) = builder.get_var(sym) {
                    let span = var.span();
                    // This is an alias pattern, so create an equality comparison in the current block with a
                    // conditional branch to the pattern failure block
                    let cond = builder.ins().eq_exact(input, value, span);
                    builder.ins().br_unless(cond, pattern_fail, &[], span);
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
                            .br_unless(is_list, pattern_fail, &[], pattern_span);
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
                };
                builder
                    .ins()
                    .br_unless(cond, pattern_fail, &[], pattern_span);
                // Since this was effectively a guard on the input, use the input value as the output of this pattern
                Ok(input)
            }
            ast::Expr::Nil(nil) => {
                let span = nil.span();
                let cond = builder.ins().eq_exact_imm(input, Immediate::Nil, span);
                builder.ins().br_unless(cond, pattern_fail, &[], span);
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
                builder.ins().br_unless(is_type, pattern_fail, &[], span);
                // Match the head
                let hd = builder.ins().head(input, span);
                self.lower_pattern(builder, *cons.head, hd, pattern_fail)?;
                // Match the tail
                let tl = builder.ins().tail(input, span);
                self.lower_pattern(builder, *cons.tail, tl, pattern_fail)?;
                Ok(input)
            }
            ast::Expr::Tuple(tuple) => {
                let span = tuple.span;
                // This gets lowered as the following sequence:
                //
                // if !is_type(Input, tuple(Arity)), then pattern_fail
                // E0 = get_element(Input, 1)
                // if E0 != Pattern[1], then pattern_fail
                // ..
                let is_type = builder
                    .ins()
                    .is_type(Type::tuple(tuple.elements.len()), input, span);
                builder.ins().br_unless(is_type, pattern_fail, &[], span);
                // Check each element
                for (index, element) in tuple.elements.iter().enumerate() {
                    let element_value = builder.ins().get_element_imm(
                        input,
                        Immediate::Integer(index.try_into().unwrap()),
                        element.span(),
                    );
                    self.lower_pattern(builder, element.clone(), element_value, pattern_fail)?;
                }
                Ok(input)
            }
            ast::Expr::Map(_map) => {
                todo!("support for maps is not implemented")
            }
            ast::Expr::Binary(_bin) => {
                todo!("support for binaries is not implemented")
            }
            ast::Expr::BinaryExpr(ref binop) if binop.op == ast::BinaryOp::Append => {
                // String prefix match
                todo!()
            }
            ast::Expr::BinaryExpr(ref binop) => {
                let span = binop.span;
                // Since only arithmetic/bitwise ops are valid here, the output term must be numeric
                let cond = match evaluator::eval_expr(&pattern, None) {
                    Ok(evaluator::Term::Number(n)) => match n {
                        Number::Integer(Integer::Small(i)) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Integer(i), span)
                        }
                        Number::Integer(big @ Integer::Big(_)) => {
                            let const_big = builder.make_constant(ConstantItem::Integer(big));
                            builder.ins().neq_exact_const(input, const_big, span)
                        }
                        Number::Float(f) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Float(f.inner()), span)
                        }
                    },
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
                builder.ins().br_if(cond, pattern_fail, &[], span);
                Ok(input)
            }
            ast::Expr::UnaryExpr(ast::UnaryExpr {
                span,
                op,
                ref operand,
            }) if op.is_valid_in_patterns() => {
                let cond = match evaluator::eval_expr(operand, None) {
                    Ok(evaluator::Term::Number(n)) => match n {
                        Number::Integer(Integer::Small(i)) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Integer(i), span)
                        }
                        Number::Integer(big @ Integer::Big(_)) => {
                            let const_big = builder.make_constant(ConstantItem::Integer(big));
                            builder.ins().neq_exact_const(input, const_big, span)
                        }
                        Number::Float(f) => {
                            builder
                                .ins()
                                .neq_exact_imm(input, Immediate::Float(f.inner()), span)
                        }
                    },
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
                builder.ins().br_if(cond, pattern_fail, &[], span);
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
