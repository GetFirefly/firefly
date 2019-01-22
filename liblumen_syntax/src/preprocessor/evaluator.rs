use failure::Error;
use rug::Integer;

use liblumen_diagnostics::ByteSpan;

use crate::lexer::{symbols, Ident, Symbol};
use crate::parser::ast::*;

use super::errors::PreprocessorError;

/// This evaluator is used for performing simple reductions
/// during preprocessing, namely for evaluating conditionals
/// in -if/-elseif directives.
///
/// As a result, the output of this function is _not_ a primitive
/// value, but rather an Expr which has been reduced to its simplest
/// form (e.g. a BinaryOp that can be evaluated at compile-time would
/// be converted into the corresponding literal representation of the
/// result of that op)
///
/// Exprs which are not able to be evaluated at compile-time will be
/// treated as errors. In particular the following constructs are supported,
/// and you can consider everything else as invalid unless explicitly noted:
///
/// - Math on constants or expressions which evaluate to constants
/// - Bit shift operations on constants or expressions which evaluate to constants
/// - Comparisons on constants or expressions which evaluate to constants
/// - The use of `++` and `--` on constant lists, or expressions which evaluate to constant lists
pub fn eval(expr: Expr) -> Result<Expr, PreprocessorError> {
    let result = match expr {
        // Nothing to be done here
        Expr::Var(_) => expr,
        Expr::Literal(_) => expr,
        Expr::Nil(_) => expr,
        Expr::FunctionName(_) => expr,
        Expr::RecordIndex(_, _, _) => expr,

        // Recursively evaluate subexpressions
        Expr::Cons(span, box head, box tail) => {
            Expr::Cons(span, Box::new(eval(head)?), Box::new(eval(tail)?))
        }
        Expr::Tuple(span, elements) => Expr::Tuple(span, eval_list(elements)?),
        Expr::Map(span, Some(box lhs), fields) => {
            Expr::Map(span, Some(Box::new(eval(lhs)?)), eval_map(fields)?)
        }
        Expr::Map(span, None, fields) => Expr::Map(span, None, eval_map(fields)?),
        Expr::Binary(span, elements) => Expr::Binary(span, eval_bin_elements(elements)?),
        Expr::RecordAccess(span, box lhs, name, field) => {
            Expr::RecordAccess(span, Box::new(eval(lhs)?), name, field)
        }
        Expr::RecordUpdate(span, box lhs, name, fields) => {
            Expr::RecordUpdate(span, Box::new(eval(lhs)?), name, eval_record(fields)?)
        }
        Expr::Record(span, name, fields) => Expr::Record(span, name, eval_record(fields)?),
        Expr::Begin(span, _) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
        Expr::Apply {
            span,
            box lhs,
            args,
        } => {
            let args = eval_list(args)?;
            let lhs = eval(lhs)?;
            match lhs {
                Expr::Literal(Literal::Atom(Ident { ref name, .. })) => match builtin(*name) {
                    None => {
                        return Err(PreprocessorError::InvalidConstExpression(span));
                    }
                    Some(fun) => match fun(args) {
                        Err(err) => return Err(PreprocessorError::BuiltinFailed(span, err)),
                        Ok(expr) => expr,
                    },
                },
                _ => return Err(PreprocessorError::InvalidConstExpression(span)),
            }
        }
        Expr::BinaryExpr {
            span,
            box lhs,
            op,
            box rhs,
        } => {
            let lhs = eval(lhs)?;
            let rhs = eval(rhs)?;
            return eval_binary_op(span, lhs, op, rhs);
        }
        Expr::UnaryExpr { span, op, box rhs } => {
            let rhs = eval(rhs)?;
            return eval_unary_op(span, op, rhs);
        }
        Expr::Remote { span, .. } => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::ListComprehension(span, _, _) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
        Expr::BinaryComprehension(span, _, _) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
        Expr::Generator(span, _, _) => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::BinaryGenerator(span, _, _) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
        Expr::Match(span, _, _) => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::If(span, _) => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::Catch(span, _) => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::Case(span, _, _) => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::Receive { span, .. } => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::Try { span, .. } => return Err(PreprocessorError::InvalidConstExpression(span)),
        Expr::Fun(Function::Named { span, .. }) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
        Expr::Fun(Function::Unnamed { span, .. }) => {
            return Err(PreprocessorError::InvalidConstExpression(span));
        }
    };

    Ok(result)
}

fn eval_list(mut exprs: Vec<Expr>) -> Result<Vec<Expr>, PreprocessorError> {
    let mut result = Vec::new();

    for expr in exprs.drain(..) {
        result.push(eval(expr)?);
    }

    Ok(result)
}

fn eval_map(mut fields: Vec<MapField>) -> Result<Vec<MapField>, PreprocessorError> {
    let mut result = Vec::new();

    for field in fields.drain(..) {
        match field {
            MapField::Assoc { span, key, value } => result.push(MapField::Assoc {
                span,
                key: eval(key)?,
                value: eval(value)?,
            }),
            MapField::Exact { span, key, value } => result.push(MapField::Exact {
                span,
                key: eval(key)?,
                value: eval(value)?,
            }),
        }
    }

    Ok(result)
}

fn eval_record(mut fields: Vec<RecordField>) -> Result<Vec<RecordField>, PreprocessorError> {
    let mut result = Vec::new();

    for field in fields.drain(..) {
        let new_field = match field {
            RecordField {
                span,
                name,
                value: Some(value),
                ty,
            } => RecordField {
                span,
                name,
                value: Some(eval(value)?),
                ty,
            },
            RecordField {
                span,
                name,
                value: None,
                ty,
            } => RecordField {
                span,
                name,
                value: None,
                ty,
            },
        };
        result.push(new_field);
    }

    Ok(result)
}

fn eval_bin_elements(
    mut elements: Vec<BinaryElement>,
) -> Result<Vec<BinaryElement>, PreprocessorError> {
    let mut result = Vec::new();

    for element in elements.drain(..) {
        let new_element = match element {
            BinaryElement {
                span,
                bit_expr,
                bit_size: Some(bit_size),
                bit_type,
            } => BinaryElement {
                span,
                bit_expr: eval(bit_expr)?,
                bit_size: Some(eval(bit_size)?),
                bit_type,
            },

            BinaryElement {
                span,
                bit_expr,
                bit_size: None,
                bit_type,
            } => BinaryElement {
                span,
                bit_expr: eval(bit_expr)?,
                bit_size: None,
                bit_type,
            },
        };

        result.push(new_element);
    }

    Ok(result)
}

fn eval_binary_op(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    match op {
        BinaryOp::OrElse | BinaryOp::AndAlso | BinaryOp::Or | BinaryOp::And => {
            eval_boolean(span, lhs, op, rhs)
        }
        BinaryOp::Equal | BinaryOp::NotEqual => eval_equality(span, lhs, op, rhs),
        BinaryOp::StrictEqual | BinaryOp::StrictNotEqual => {
            eval_strict_equality(span, lhs, op, rhs)
        }
        BinaryOp::Lte | BinaryOp::Lt | BinaryOp::Gte | BinaryOp::Gt => {
            eval_comparison(span, lhs, op, rhs)
        }
        BinaryOp::Add
        | BinaryOp::Sub
        | BinaryOp::Multiply
        | BinaryOp::Divide
        | BinaryOp::Div
        | BinaryOp::Rem => eval_arith(span, lhs, op, rhs),
        BinaryOp::Bor
        | BinaryOp::Bxor
        | BinaryOp::Xor
        | BinaryOp::Band
        | BinaryOp::Bsl
        | BinaryOp::Bsr => eval_shift(span, lhs, op, rhs),
        _ => return Err(PreprocessorError::InvalidConstExpression(span)),
    }
}

fn eval_unary_op(span: ByteSpan, op: UnaryOp, rhs: Expr) -> Result<Expr, PreprocessorError> {
    let expr = match op {
        UnaryOp::Plus => match rhs {
            Expr::Literal(Literal::Integer(span, i)) if i < 0 => {
                Expr::Literal(Literal::Integer(span, i * -1))
            }
            Expr::Literal(Literal::Integer(_, _)) => rhs,
            Expr::Literal(Literal::BigInteger(span, i)) => {
                if i < 0 {
                    Expr::Literal(Literal::BigInteger(span, i * -1))
                } else {
                    Expr::Literal(Literal::BigInteger(span, i))
                }
            }
            Expr::Literal(Literal::Float(span, i)) if i < 0.0 => {
                Expr::Literal(Literal::Float(span, i * -1.0))
            }
            Expr::Literal(Literal::Float(_, _)) => rhs,
            _ => return Err(PreprocessorError::InvalidConstExpression(span)),
        },
        UnaryOp::Minus => match rhs {
            Expr::Literal(Literal::Integer(span, i)) if i > 0 => {
                Expr::Literal(Literal::Integer(span, i * -1))
            }
            Expr::Literal(Literal::Integer(_, _)) => rhs,
            Expr::Literal(Literal::BigInteger(span, i)) => {
                if i > 0 {
                    Expr::Literal(Literal::BigInteger(span, i * -1))
                } else {
                    Expr::Literal(Literal::BigInteger(span, i))
                }
            }
            Expr::Literal(Literal::Float(span, i)) if i > 0.0 => {
                Expr::Literal(Literal::Float(span, i * -1.0))
            }
            Expr::Literal(Literal::Float(_, _)) => rhs,
            _ => return Err(PreprocessorError::InvalidConstExpression(span)),
        },
        UnaryOp::Bnot => match rhs {
            Expr::Literal(Literal::Integer(span, i)) => Expr::Literal(Literal::Integer(span, !i)),
            Expr::Literal(Literal::BigInteger(span, i)) => {
                Expr::Literal(Literal::BigInteger(span, !i))
            }
            _ => return Err(PreprocessorError::InvalidConstExpression(span)),
        },
        UnaryOp::Not => match rhs {
            Expr::Literal(Literal::Atom(Ident { name, span })) if name == symbols::True => {
                Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                }))
            }
            Expr::Literal(Literal::Atom(Ident { name, span })) if name == symbols::False => {
                Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                }))
            }
            _ => return Err(PreprocessorError::InvalidConstExpression(span)),
        },
    };
    Ok(expr)
}

fn eval_boolean(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    if !is_boolean(&lhs) || !is_boolean(&rhs) {
        return Err(PreprocessorError::InvalidConstExpression(span));
    }
    let left = is_true(&lhs);
    let right = is_true(&rhs);

    match op {
        BinaryOp::Xor => {
            if (left != right) && (left || right) {
                return Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })));
            }
            return Ok(Expr::Literal(Literal::Atom(Ident {
                name: symbols::False,
                span,
            })));
        }
        BinaryOp::OrElse | BinaryOp::Or => {
            if left || right {
                return Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })));
            } else {
                return Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })));
            }
        }
        BinaryOp::AndAlso | BinaryOp::And => {
            if left && right {
                return Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })));
            } else {
                return Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })));
            }
        }
        _ => unreachable!(),
    }
}

fn eval_equality(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    if is_number(&lhs) && is_number(&rhs) {
        eval_numeric_equality(span, lhs, op, rhs)
    } else {
        match op {
            BinaryOp::Equal => {
                if lhs == rhs {
                    Ok(Expr::Literal(Literal::Atom(Ident {
                        name: symbols::True,
                        span,
                    })))
                } else {
                    Ok(Expr::Literal(Literal::Atom(Ident {
                        name: symbols::False,
                        span,
                    })))
                }
            }
            BinaryOp::NotEqual => {
                if lhs != rhs {
                    Ok(Expr::Literal(Literal::Atom(Ident {
                        name: symbols::True,
                        span,
                    })))
                } else {
                    Ok(Expr::Literal(Literal::Atom(Ident {
                        name: symbols::False,
                        span,
                    })))
                }
            }
            _ => unreachable!(),
        }
    }
}

fn eval_strict_equality(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    match op {
        BinaryOp::StrictEqual => {
            if lhs == rhs {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })))
            } else {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })))
            }
        }
        BinaryOp::StrictNotEqual => {
            if lhs != rhs {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })))
            } else {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })))
            }
        }
        _ => unreachable!(),
    }
}

fn eval_numeric_equality(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    let result = match (lhs, rhs) {
        (Expr::Literal(Literal::Integer(_, x)), Expr::Literal(Literal::Integer(_, y))) => {
            match op {
                BinaryOp::Equal if x == y => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })),
                BinaryOp::NotEqual if x != y => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })),
                BinaryOp::Equal => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })),
                BinaryOp::NotEqual => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })),
                _ => unreachable!(),
            }
        }
        (Expr::Literal(Literal::BigInteger(_, x)), Expr::Literal(Literal::BigInteger(_, y))) => {
            match op {
                BinaryOp::Equal if x == y => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })),
                BinaryOp::NotEqual if x != y => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })),
                BinaryOp::Equal => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })),
                BinaryOp::NotEqual => Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })),
                _ => unreachable!(),
            }
        }
        (Expr::Literal(Literal::Float(_, x)), Expr::Literal(Literal::Float(_, y))) => match op {
            BinaryOp::Equal if x == y => Expr::Literal(Literal::Atom(Ident {
                name: symbols::True,
                span,
            })),
            BinaryOp::NotEqual if x != y => Expr::Literal(Literal::Atom(Ident {
                name: symbols::True,
                span,
            })),
            BinaryOp::Equal => Expr::Literal(Literal::Atom(Ident {
                name: symbols::False,
                span,
            })),
            BinaryOp::NotEqual => Expr::Literal(Literal::Atom(Ident {
                name: symbols::False,
                span,
            })),
            _ => unreachable!(),
        },
        (
            Expr::Literal(Literal::Integer(xspan, x)),
            rhs @ Expr::Literal(Literal::BigInteger(_, _)),
        ) => {
            return eval_numeric_equality(
                span,
                Expr::Literal(Literal::BigInteger(xspan, Integer::from(x))),
                op,
                rhs,
            );
        }

        (Expr::Literal(Literal::Integer(xspan, x)), rhs @ Expr::Literal(Literal::Float(_, _))) => {
            return eval_numeric_equality(
                span,
                Expr::Literal(Literal::Float(xspan, x as f64)),
                op,
                rhs,
            );
        }

        (
            lhs @ Expr::Literal(Literal::BigInteger(_, _)),
            Expr::Literal(Literal::Integer(yspan, y)),
        ) => {
            return eval_numeric_equality(
                span,
                lhs,
                op,
                Expr::Literal(Literal::BigInteger(yspan, Integer::from(y))),
            );
        }

        (lhs @ Expr::Literal(Literal::Float(_, _)), Expr::Literal(Literal::Integer(yspan, y))) => {
            return eval_numeric_equality(
                span,
                lhs,
                op,
                Expr::Literal(Literal::Float(yspan, y as f64)),
            );
        }

        _ => return Err(PreprocessorError::InvalidConstExpression(span)),
    };

    Ok(result)
}

fn eval_comparison(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    match op {
        BinaryOp::Lt | BinaryOp::Lte => {
            if lhs < rhs {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })))
            } else if op == BinaryOp::Lte {
                eval_equality(span, lhs, BinaryOp::Equal, rhs)
            } else {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })))
            }
        }
        BinaryOp::Gt | BinaryOp::Gte => {
            if lhs > rhs {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::True,
                    span,
                })))
            } else if op == BinaryOp::Gte {
                eval_equality(span, lhs, BinaryOp::Equal, rhs)
            } else {
                Ok(Expr::Literal(Literal::Atom(Ident {
                    name: symbols::False,
                    span,
                })))
            }
        }
        _ => unreachable!(),
    }
}

fn eval_arith(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    if is_number(&lhs) && is_number(&rhs) {
        let result = match (lhs, rhs) {
            // Types match
            (Expr::Literal(Literal::Integer(_, x)), Expr::Literal(Literal::Integer(_, y))) => {
                eval_op_int(span, x, op, y)?
            }
            (
                Expr::Literal(Literal::BigInteger(_, x)),
                Expr::Literal(Literal::BigInteger(_, y)),
            ) => eval_op_bigint(span, x, op, y)?,
            (Expr::Literal(Literal::Float(_, x)), Expr::Literal(Literal::Float(_, y))) => {
                eval_op_float(span, x, op, y)?
            }

            // Coerce to BigInt
            (Expr::Literal(Literal::Integer(_, x)), Expr::Literal(Literal::BigInteger(_, y))) => {
                eval_op_bigint(span, Integer::from(x), op, y)?
            }
            (Expr::Literal(Literal::BigInteger(_, x)), Expr::Literal(Literal::Integer(_, y))) => {
                eval_op_bigint(span, x, op, Integer::from(y))?
            }

            // Coerce to float
            (Expr::Literal(Literal::Integer(_, x)), Expr::Literal(Literal::Float(_, y))) => {
                eval_op_float(span, x as f64, op, y)?
            }
            (Expr::Literal(Literal::Float(_, x)), Expr::Literal(Literal::Integer(_, y))) => {
                eval_op_float(span, x, op, y as f64)?
            }

            _ => return Err(PreprocessorError::InvalidConstExpression(span)),
        };
        Ok(result)
    } else {
        return Err(PreprocessorError::InvalidConstExpression(span));
    }
}

fn eval_op_int(span: ByteSpan, x: i64, op: BinaryOp, y: i64) -> Result<Expr, PreprocessorError> {
    let result = match op {
        BinaryOp::Add => Expr::Literal(Literal::Integer(span, x + y)),
        BinaryOp::Sub => Expr::Literal(Literal::Integer(span, x - y)),
        BinaryOp::Multiply => Expr::Literal(Literal::Integer(span, x * y)),
        BinaryOp::Divide if y == 0 => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Divide => Expr::Literal(Literal::Float(span, (x as f64) / (y as f64))),
        BinaryOp::Div if y == 0 => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Div => Expr::Literal(Literal::Integer(span, x / y)),
        BinaryOp::Rem => Expr::Literal(Literal::Integer(span, x % y)),
        _ => unreachable!(),
    };
    Ok(result)
}

fn eval_op_bigint(
    span: ByteSpan,
    x: Integer,
    op: BinaryOp,
    y: Integer,
) -> Result<Expr, PreprocessorError> {
    let zero = Integer::new();
    let result = match op {
        BinaryOp::Add => Expr::Literal(Literal::BigInteger(span, x + y)),
        BinaryOp::Sub => Expr::Literal(Literal::BigInteger(span, x - y)),
        BinaryOp::Multiply => Expr::Literal(Literal::BigInteger(span, x * y)),
        BinaryOp::Divide => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Div if y == zero => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Div => Expr::Literal(Literal::BigInteger(span, x / y)),
        BinaryOp::Rem => Expr::Literal(Literal::BigInteger(span, x % y)),
        _ => unreachable!(),
    };
    Ok(result)
}

fn eval_op_float(span: ByteSpan, x: f64, op: BinaryOp, y: f64) -> Result<Expr, PreprocessorError> {
    match op {
        BinaryOp::Add => Ok(Expr::Literal(Literal::Float(span, x + y))),
        BinaryOp::Sub => Ok(Expr::Literal(Literal::Float(span, x - y))),
        BinaryOp::Multiply => Ok(Expr::Literal(Literal::Float(span, x * y))),
        BinaryOp::Divide if y == 0.0 => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Divide => Ok(Expr::Literal(Literal::Float(span, x / y))),
        BinaryOp::Div => return Err(PreprocessorError::InvalidConstExpression(span)),
        BinaryOp::Rem => return Err(PreprocessorError::InvalidConstExpression(span)),
        _ => unreachable!(),
    }
}

fn eval_shift(
    span: ByteSpan,
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
) -> Result<Expr, PreprocessorError> {
    match (lhs, rhs) {
        (Expr::Literal(Literal::Integer(_, x)), Expr::Literal(Literal::Integer(_, y))) => {
            let result = match op {
                BinaryOp::Bor => Expr::Literal(Literal::Integer(span, x | y)),
                BinaryOp::Bxor => Expr::Literal(Literal::Integer(span, x ^ y)),
                BinaryOp::Band => Expr::Literal(Literal::Integer(span, x & y)),
                BinaryOp::Bsl => Expr::Literal(Literal::Integer(span, x << y)),
                BinaryOp::Bsr => Expr::Literal(Literal::Integer(span, x >> y)),
                _ => unreachable!(),
            };
            Ok(result)
        }
        _ => return Err(PreprocessorError::InvalidConstExpression(span)),
    }
}

fn is_number(e: &Expr) -> bool {
    match *e {
        Expr::Literal(Literal::Integer(_, _)) => true,
        Expr::Literal(Literal::BigInteger(_, _)) => true,
        Expr::Literal(Literal::Float(_, _)) => true,
        _ => false,
    }
}

fn is_boolean(e: &Expr) -> bool {
    match *e {
        Expr::Literal(Literal::Atom(Ident { ref name, .. })) => {
            if *name == symbols::True || *name == symbols::False {
                return true;
            }
            false
        }
        _ => false,
    }
}

fn is_true(e: &Expr) -> bool {
    match *e {
        Expr::Literal(Literal::Atom(Ident { ref name, .. })) if *name == symbols::True => true,
        _ => false,
    }
}

fn builtin(_name: Symbol) -> Option<&'static fn(Vec<Expr>) -> Result<Expr, Error>> {
    None
}
