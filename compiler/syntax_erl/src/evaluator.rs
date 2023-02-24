use std::cmp::Ordering;
use std::collections::BTreeMap;

use firefly_binary::{BinaryEntrySpecifier, BitVec, Bitstring};
use firefly_diagnostics::{Diagnostic, Label, SourceSpan, Spanned, ToDiagnostic};
use firefly_intern::{symbols, Ident};
use firefly_number::{f16, Int, Number, ToPrimitive};
use firefly_syntax_base::{BinaryOp, UnaryOp};

use crate::ast::{self, BinaryElement, Expr, Literal};

#[derive(Debug, thiserror::Error, Spanned)]
pub enum EvalError {
    #[error("invalid const expression")]
    InvalidConstExpression {
        #[span]
        span: SourceSpan,
    },

    #[error("invalid float: {ty}")]
    FloatError {
        ty: firefly_number::FloatError,
        #[span]
        span: SourceSpan,
    },

    #[error("integer division requires both operands to be integers")]
    InvalidDivOperand {
        #[span]
        span: SourceSpan,
    },

    #[error("expression evaluated to division by zero")]
    DivisionByZero {
        #[span]
        span: SourceSpan,
    },

    #[error("bitwise operators requires all operands to be integers")]
    InvalidBitwiseOperand {
        #[span]
        span: SourceSpan,
    },

    #[error("attempted too large bitshift")]
    TooLargeShift {
        #[span]
        span: SourceSpan,
    },

    #[error("no record with name")]
    NoRecord {
        #[span]
        span: SourceSpan,
    },

    #[error("field doesn't exist in record")]
    NoRecordField {
        #[span]
        span: SourceSpan,
    },

    #[error("map does not contain key `{key}`")]
    InvalidMapKey {
        #[span]
        span: SourceSpan,
        key: Literal,
    },
}
impl ToDiagnostic for EvalError {
    fn to_diagnostic(self) -> Diagnostic {
        let msg = self.to_string();
        match self {
            EvalError::InvalidConstExpression { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message("expression not evaluable to constant")]),
            EvalError::FloatError { span, .. } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::InvalidDivOperand { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::DivisionByZero { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::InvalidBitwiseOperand { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::TooLargeShift { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::NoRecord { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::NoRecordField { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
            EvalError::InvalidMapKey { span, .. } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)]),
        }
    }
}

#[allow(dead_code)]
pub enum ResolveRecordIndexError {
    NoRecord,
    NoField,
}

#[derive(Debug, Clone, Default)]
pub struct Bindings(BTreeMap<Ident, Expr>);
impl Bindings {
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[allow(dead_code)]
    pub fn get(&self, name: &Ident) -> Option<&Expr> {
        self.0.get(name)
    }

    #[allow(dead_code)]
    pub fn add(&mut self, name: Ident, value: Expr) -> bool {
        self.0.insert(name, value).is_some()
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, name: &Ident) {
        self.0.remove(name);
    }

    #[allow(dead_code)]
    pub fn merge(&self, other: &Self) -> Result<Self, ()> {
        use std::collections::btree_map::Entry;

        let mut merged = self.0.clone();
        for (key, value) in other.0.iter() {
            match merged.entry(*key) {
                Entry::Vacant(entry) => {
                    entry.insert(value.clone());
                }
                Entry::Occupied(entry) => {
                    if entry.get() != value {
                        return Err(());
                    }
                }
            }
        }

        Ok(Self(merged))
    }
}

pub fn eval_expr(
    expr: &Expr,
    resolve_record_index: Option<&dyn Fn(Ident, Ident) -> Result<usize, ResolveRecordIndexError>>,
) -> Result<Literal, EvalError> {
    let span = expr.span();

    let res = match expr {
        Expr::Literal(lit) => lit.clone(),
        Expr::Cons(cons) => {
            let head = eval_expr(&cons.head, resolve_record_index)?;
            let tail = eval_expr(&cons.tail, resolve_record_index)?;
            Literal::Cons(cons.span, Box::new(head), Box::new(tail))
        }
        Expr::Tuple(tup) => Literal::Tuple(
            tup.span,
            tup.elements
                .iter()
                .map(|e| eval_expr(e, resolve_record_index))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Map(ast::Map { span, fields }) => {
            let mut stored: BTreeMap<Literal, Literal> = BTreeMap::new();
            for field in fields.iter() {
                match field {
                    ast::MapField::Assoc {
                        ref key, ref value, ..
                    } => {
                        let key = eval_expr(key, resolve_record_index)?;
                        let value = eval_expr(value, resolve_record_index)?;
                        stored.insert(key, value);
                    }
                    ast::MapField::Exact {
                        span,
                        ref key,
                        ref value,
                    } => {
                        let key = eval_expr(key, resolve_record_index)?;
                        let value = eval_expr(value, resolve_record_index)?;
                        if let Err(err) = stored.try_insert(key, value) {
                            return Err(EvalError::InvalidMapKey {
                                span: *span,
                                key: err.entry.key().clone(),
                            });
                        }
                    }
                }
            }
            Literal::Map(*span, stored)
        }
        Expr::Binary(_) => unimplemented!(),
        Expr::Record(_) => unimplemented!(),
        Expr::RecordIndex(rec_idx) if resolve_record_index.is_some() => {
            let span = rec_idx.span;
            match resolve_record_index.unwrap()(rec_idx.name, rec_idx.field) {
                Ok(index) => Literal::Integer(span, index.into()),
                Err(ResolveRecordIndexError::NoRecord) => Err(EvalError::NoRecord { span })?,
                Err(ResolveRecordIndexError::NoField) => Err(EvalError::NoRecordField { span })?,
            }
        }

        Expr::BinaryExpr(bin_expr) => {
            use BinaryOp as B;

            let span = bin_expr.span;
            let lhs = eval_expr(&bin_expr.lhs, resolve_record_index)?;
            let rhs = eval_expr(&bin_expr.rhs, resolve_record_index)?;

            match (bin_expr.op, lhs, rhs) {
                (B::Add, lhs, rhs) => {
                    let lhs: Number = lhs.try_into().unwrap();
                    let rhs: Number = rhs.try_into().unwrap();
                    (lhs + rhs)
                        .map_err(|ty| EvalError::FloatError { span, ty })?
                        .into()
                }
                (B::Sub, lhs, rhs) => {
                    let lhs: Number = lhs.try_into().unwrap();
                    let rhs: Number = rhs.try_into().unwrap();
                    (lhs - rhs)
                        .map_err(|ty| EvalError::FloatError { span, ty })?
                        .into()
                }
                (B::Multiply, lhs, rhs) => {
                    let lhs: Number = lhs.try_into().unwrap();
                    let rhs: Number = rhs.try_into().unwrap();
                    (lhs * rhs)
                        .map_err(|ty| EvalError::FloatError { span, ty })?
                        .into()
                }
                (B::Divide, lhs, rhs) => {
                    let rhs: Number = rhs.try_into().unwrap();
                    let lhs: Number = lhs.try_into().unwrap();
                    (lhs / rhs)
                        .map_err(|_| EvalError::DivisionByZero { span })?
                        .into()
                }
                (B::Div, Literal::Integer(_, l), Literal::Integer(_, r)) => Literal::Integer(
                    span,
                    (l / r).map_err(|_| EvalError::DivisionByZero { span })?,
                ),
                (B::Div, _, _) => Err(EvalError::InvalidDivOperand { span })?,

                (B::Bor, Literal::Integer(_, l), Literal::Integer(_, r)) => {
                    Literal::Integer(span, l | r).into()
                }
                (B::Band, Literal::Integer(_, l), Literal::Integer(_, r)) => {
                    Literal::Integer(span, l & r).into()
                }
                (B::Bxor, Literal::Integer(_, l), Literal::Integer(_, r)) => {
                    Literal::Integer(span, l ^ r).into()
                }
                (B::Bsl, Literal::Integer(_, l), Literal::Integer(_, r)) => {
                    let shift = r.to_u32().ok_or(EvalError::TooLargeShift { span })?;
                    Literal::Integer(span, l << shift).into()
                }
                (B::Bsr, Literal::Integer(_, l), Literal::Integer(_, r)) => {
                    let shift = r.to_u32().ok_or(EvalError::TooLargeShift { span })?;
                    Literal::Integer(span, l >> shift).into()
                }
                (B::Bor | B::Band | B::Bxor | B::Bsl | B::Bsr, _, _) => {
                    return Err(EvalError::InvalidBitwiseOperand { span });
                }

                (B::Lt, l, r) => (l < r).into(),
                (B::Lte, l, r) => (l <= r).into(),
                (B::Gt, l, r) => (l > r).into(),
                (B::Gte, l, r) => (l >= r).into(),
                (B::Equal, l, r) => (l.cmp(&r) == Ordering::Equal).into(),
                (B::NotEqual, l, r) => (l.cmp(&r) != Ordering::Equal).into(),
                (B::StrictEqual, l, r) => l.eq(&r).into(),
                (B::StrictNotEqual, l, r) => (!l.eq(&r)).into(),

                // [] op []
                // [] op []
                (B::Append, Literal::Nil(_), nil @ Literal::Nil(_))
                | (B::Remove, Literal::Nil(_), nil @ Literal::Nil(_)) => nil,
                // [] op [_ | _]
                // [] op "foo"
                (
                    B::Append,
                    Literal::Nil(_),
                    cons @ (Literal::Cons(_, _, _) | Literal::String(_)),
                ) => cons,
                (B::Remove, nil @ Literal::Nil(_), Literal::Cons(_, _, _) | Literal::String(_)) => {
                    nil
                }
                // [_ | _] op []
                (B::Append, cons @ Literal::Cons(_, _, _), Literal::Nil(_)) => cons,
                (B::Remove, cons @ Literal::Cons(_, _, _), Literal::Nil(_)) => cons,
                // [_ | _] ++ [_ | _]
                // [_ | _] ++ "foo"
                (B::Append, l @ Literal::Cons(_, _, _), r) => {
                    let rspan = r.span();
                    if let Ok(mut xs) = l.as_proper_list() {
                        if let Ok(mut tail) = r.as_proper_list() {
                            xs.append(&mut tail);

                            xs.drain(..).rfold(Literal::Nil(rspan), |lit, tail| {
                                Literal::Cons(lit.span(), Box::new(lit), Box::new(tail))
                            })
                        } else {
                            xs.drain(..).rfold(r, |lit, tail| {
                                Literal::Cons(lit.span(), Box::new(lit), Box::new(tail))
                            })
                        }
                    } else {
                        return Err(EvalError::InvalidConstExpression { span });
                    }
                }
                // [_ | _] -- [_ | _]
                // [_ | _] -- "foo"
                (B::Remove, l @ Literal::Cons(_, _, _), r) => {
                    // Both sides must be proper lists
                    let mut ls = l
                        .as_proper_list()
                        .map_err(|_| EvalError::InvalidConstExpression { span })?;
                    let mut rs = r
                        .as_proper_list()
                        .map_err(|_| EvalError::InvalidConstExpression { span })?;

                    // For each element in the right-hand list, remove the first occurrance in the left-hand list
                    for element in rs.drain(..) {
                        if let Some(idx) = ls.iter().position(|l| l.eq(&element)) {
                            ls.remove(idx);
                        }
                    }

                    // Construct the result list from the remaining elements of the left-hand list
                    ls.drain(..).rfold(Literal::Nil(span), |lit, tail| {
                        Literal::Cons(lit.span(), Box::new(lit), Box::new(tail))
                    })
                }
                // "foo" op []
                // "foo" op [_ | _]
                // "foo" op "bar"
                (op @ (B::Append | B::Remove), l @ Literal::String(_), r) => {
                    let mut ls = l.as_proper_list().unwrap();
                    let l = ls.drain(..).rfold(Literal::Nil(span), |lit, tail| {
                        Literal::Cons(lit.span(), Box::new(lit), Box::new(tail))
                    });
                    let expr = Expr::BinaryExpr(ast::BinaryExpr {
                        span,
                        op,
                        lhs: Box::new(Expr::Literal(l)),
                        rhs: Box::new(Expr::Literal(r)),
                    });
                    eval_expr(&expr, resolve_record_index)?
                }
                (B::Append | B::Remove, _, _) => {
                    return Err(EvalError::InvalidConstExpression { span })
                }

                _ => return Err(EvalError::InvalidConstExpression { span }),
            }
        }

        Expr::UnaryExpr(un_expr) => {
            use UnaryOp as U;

            let operand = eval_expr(&un_expr.operand, resolve_record_index)?;

            match (un_expr.op, operand) {
                (U::Plus, lit) => {
                    let n: Number = lit
                        .try_into()
                        .map_err(|_| EvalError::InvalidConstExpression { span })?;
                    n.abs().into()
                }
                (U::Minus, lit) => {
                    let n: Number = lit
                        .try_into()
                        .map_err(|_| EvalError::InvalidConstExpression { span })?;
                    (-n).into()
                }
                (U::Bnot, Literal::Integer(span, i)) => Literal::Integer(span, !i),
                (U::Bnot, _) => Err(EvalError::InvalidBitwiseOperand { span })?,

                (U::Not, Literal::Atom(sym)) if sym == symbols::True => false.into(),
                (U::Not, Literal::Atom(sym)) if sym == symbols::False => true.into(),

                _ => Err(EvalError::InvalidConstExpression { span })?,
            }
        }

        _ => Err(EvalError::InvalidConstExpression { span })?,
    };

    Ok(res)
}

pub fn expr_grp<F>(fields: &[BinaryElement], bindings: &mut Bindings, eval: F) -> Result<BitVec, ()>
where
    F: Fn(Expr, &mut Bindings) -> Result<Expr, ()>,
{
    let mut bin = BitVec::new();
    for field in fields {
        eval_field(field, &mut bin, bindings, &eval)?;
    }
    Ok(bin)
}

fn eval_field<F>(
    field: &BinaryElement,
    bin: &mut BitVec,
    bindings: &mut Bindings,
    eval: F,
) -> Result<(), ()>
where
    F: Fn(Expr, &mut Bindings) -> Result<Expr, ()>,
{
    let value = eval(field.bit_expr.clone(), bindings)?;
    let size = match field.bit_size.clone() {
        None => None,
        Some(bit_size) => match eval(bit_size, bindings)? {
            Expr::Literal(Literal::Integer(_, i)) => Some(i.to_usize().ok_or(())?),
            _ => return Err(()),
        },
    };
    let spec = field.specifier.unwrap_or_default();
    match spec {
        BinaryEntrySpecifier::Integer {
            signed,
            endianness,
            unit,
        } => {
            let unit = unit as usize;
            let size = size.unwrap_or(8);
            match value {
                Expr::Literal(Literal::String(s)) => {
                    if spec == BinaryEntrySpecifier::DEFAULT {
                        bin.push_str(s.as_str().get());
                    } else {
                        for c in s.as_str().get().chars() {
                            bin.push_ap_number(c as u32, size * unit, endianness);
                        }
                    }
                }
                value => {
                    let integer = match value {
                        Expr::Literal(Literal::Char(_, c)) => Int::new((c as u32) as i64),
                        Expr::Literal(Literal::Integer(_, i)) => i,
                        Expr::Literal(Literal::Float(_, f)) => f.to_integer(),
                        _ => return Err(()),
                    };
                    match integer {
                        Int::Small(i) if signed => bin.push_ap_number(i, size * unit, endianness),
                        Int::Small(i) => bin.push_ap_number(i as u64, size * unit, endianness),
                        Int::Big(ref i) => bin.push_ap_bigint(i, size * unit, signed, endianness),
                    }
                }
            }
        }
        BinaryEntrySpecifier::Float { endianness, .. } => {
            let size = match size.unwrap_or(64) {
                i @ (16 | 32 | 64) => i,
                _ => return Err(()),
            };
            match value {
                Expr::Literal(Literal::Char(_, c)) => match size {
                    16 => {
                        let f = f16::from_f32((c as u32) as f32);
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    32 => {
                        let f = (c as u32) as f32;
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    64 => bin.push_number((c as u64) as f64, endianness),
                    _ => return Err(()),
                },
                Expr::Literal(Literal::Integer(_, Int::Small(i))) => match size {
                    16 => {
                        let i: i32 = i.try_into().map_err(|_| ())?;
                        let f = f16::from_f32(i as f32);
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    32 => {
                        let i: i32 = i.try_into().map_err(|_| ())?;
                        let f = i as f32;
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    64 => {
                        let f = i as f64;
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    _ => return Err(()),
                },
                Expr::Literal(Literal::Float(_, f)) => match size {
                    16 => {
                        let f: f16 = f.into();
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    32 => {
                        let f: f32 = f.into();
                        if f.is_normal() {
                            bin.push_number(f, endianness);
                        } else {
                            return Err(());
                        }
                    }
                    64 => bin.push_number(f.raw(), endianness),
                    _ => return Err(()),
                },
                _ => return Err(()),
            }
        }
        BinaryEntrySpecifier::Binary { unit } => {
            let unit = unit as usize;
            if let Some(size) = size {
                // Use N * Unit bits of the value
                let bitsize = size * unit;
                match value {
                    Expr::Literal(Literal::Binary(_, b)) => {
                        let bytes = unsafe { b.as_bytes_unchecked() };
                        bin.push_bits(bytes, bitsize);
                    }
                    _ => return Err(()),
                }
            } else {
                // Use the entire value by default
                match value {
                    Expr::Literal(Literal::Binary(_, b)) => {
                        bin.concat(&b);
                    }
                    _ => return Err(()),
                }
            }
        }
        BinaryEntrySpecifier::Utf8 => {
            assert!(size.is_none());
            let codepoint = match value {
                Expr::Literal(Literal::Char(_, c)) => c,
                Expr::Literal(Literal::Integer(_, Int::Small(i))) => {
                    let i: u32 = i.try_into().map_err(|_| ())?;
                    char::from_u32(i).ok_or(())?
                }
                _ => return Err(()),
            };
            bin.push_utf8(codepoint);
        }
        BinaryEntrySpecifier::Utf16 { endianness } => {
            assert!(size.is_none());
            match value {
                Expr::Literal(Literal::Char(_, c)) => bin.push_utf16(c, endianness),
                Expr::Literal(Literal::Integer(_, Int::Small(i))) => {
                    let i: u16 = i.try_into().map_err(|_| ())?;
                    for r in std::char::decode_utf16(core::iter::once(i)) {
                        let c = r.map_err(|_| ())?;
                        bin.push_utf16(c, endianness);
                    }
                }
                _ => return Err(()),
            }
        }
        BinaryEntrySpecifier::Utf32 { endianness } => {
            assert!(size.is_none());
            let codepoint = match value {
                Expr::Literal(Literal::Char(_, c)) => c,
                Expr::Literal(Literal::Integer(_, Int::Small(i))) => {
                    let i: u32 = i.try_into().map_err(|_| ())?;
                    char::from_u32(i).ok_or(())?
                }
                _ => return Err(()),
            };
            bin.push_utf32(codepoint, endianness);
        }
    }

    Ok(())
}
