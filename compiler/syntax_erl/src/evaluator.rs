#![allow(dead_code, unused_variables)]
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::hash::Hash;

use crate::ast::{BinaryOp, Expr, Literal, UnaryOp};
use crate::lexer::symbols;

use liblumen_diagnostics::{Diagnostic, Label, SourceSpan, ToDiagnostic};
use liblumen_intern::{Ident, Symbol};
use liblumen_number::{Number, ToPrimitive};

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("invalid const expression")]
    InvalidConstExpression { span: SourceSpan },

    #[error("invalid float: {source}")]
    FloatError {
        source: liblumen_number::FloatError,
        span: SourceSpan,
    },

    #[error("integer division requires both operands to be integers")]
    InvalidDivOperand { span: SourceSpan },

    #[error("expression evaluated to division by zero")]
    DivisionByZero { span: SourceSpan },

    #[error("bitwise operators requires all operands to be integers")]
    InvalidBitwiseOperand { span: SourceSpan },

    #[error("attempted too large bitshift")]
    TooLargeShift { span: SourceSpan },

    #[error("no record with name")]
    NoRecord { span: SourceSpan },

    #[error("field doesn't exist in record")]
    NoRecordField { span: SourceSpan },
}
impl EvalError {
    pub fn span(&self) -> SourceSpan {
        match self {
            Self::InvalidConstExpression { span }
            | Self::FloatError { span, .. }
            | Self::InvalidDivOperand { span }
            | Self::DivisionByZero { span }
            | Self::InvalidBitwiseOperand { span }
            | Self::TooLargeShift { span }
            | Self::NoRecord { span }
            | Self::NoRecordField { span } => *span,
        }
    }
}
impl ToDiagnostic for EvalError {
    fn to_diagnostic(&self) -> Diagnostic {
        let msg = self.to_string();
        match self {
            EvalError::InvalidConstExpression { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)
                    .with_message("expression not evaluable to constant")]),
            EvalError::FloatError { span, .. } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::InvalidDivOperand { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::DivisionByZero { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::InvalidBitwiseOperand { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::TooLargeShift { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::NoRecord { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
            EvalError::NoRecordField { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)]),
        }
    }
}

#[derive(Clone, Hash)]
pub enum Term {
    Cons(Box<Term>, Box<Term>),
    Nil,
    Map(BTreeMap<Term, Term>),
    Tuple(Vec<Term>),
    Atom(Symbol),
    Number(Number),
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        self.equals(other, false)
    }
}
impl Eq for Term {}

impl Ord for Term {
    // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string
    fn cmp(&self, other: &Term) -> Ordering {
        match (self, other) {
            (Term::Number(l), Term::Number(r)) => l.cmp(r),
            (Term::Number(_), _) => Ordering::Less,

            (Term::Atom(_), Term::Number(_)) => Ordering::Greater,
            (Term::Atom(l), Term::Atom(r)) => l.cmp(r),
            (Term::Atom(_), _) => Ordering::Less,

            (Term::Tuple(_), Term::Number(_) | Term::Atom(_)) => Ordering::Greater,
            (Term::Tuple(l), Term::Tuple(r)) => l.cmp(r),
            (Term::Tuple(_), _) => Ordering::Less,

            (Term::Map(_), Term::Number(_) | Term::Atom(_) | Term::Tuple(_)) => Ordering::Greater,
            (Term::Map(l), Term::Map(r)) => l.cmp(r),
            (Term::Map(_), _) => Ordering::Less,

            (Term::Nil, Term::Number(_) | Term::Atom(_) | Term::Tuple(_) | Term::Map(_)) => {
                Ordering::Greater
            }
            (Term::Nil, Term::Nil) => Ordering::Equal,
            (Term::Nil, _) => Ordering::Less,

            (
                Term::Cons(_, _),
                Term::Number(_) | Term::Atom(_) | Term::Tuple(_) | Term::Map(_) | Term::Nil,
            ) => Ordering::Greater,
            (Term::Cons(lh, lt), Term::Cons(rh, rt)) => (lh, lt).cmp(&(rh, rt)),
        }
    }
}
impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Term) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<Number> for Term {
    fn from(num: Number) -> Term {
        Term::Number(num)
    }
}
impl From<bool> for Term {
    fn from(cond: bool) -> Term {
        if cond {
            Term::Atom(symbols::True)
        } else {
            Term::Atom(symbols::False)
        }
    }
}

impl Term {
    pub fn equals(&self, rhs: &Term, exact: bool) -> bool {
        match (self, rhs) {
            (Term::Atom(l), Term::Atom(r)) => l == r,
            (Term::Number(l), Term::Number(r)) => l.equals(r, exact),
            (Term::Tuple(l), Term::Tuple(r)) => l == r,
            (Term::Map(l), Term::Map(r)) => l == r,
            (Term::Nil, Term::Nil) => true,
            (Term::Cons(lh, lt), Term::Cons(rh, rt)) => lh == rh && lt == rt,
            _ => false,
        }
    }
}

pub enum ResolveRecordIndexError {
    NoRecord,
    NoField,
}

pub fn eval_expr(
    expr: &Expr,
    resolve_record_index: Option<&dyn Fn(Ident, Ident) -> Result<usize, ResolveRecordIndexError>>,
) -> Result<Term, EvalError> {
    let span = expr.span();
    let invalid_expr = EvalError::InvalidConstExpression { span };

    let res = match expr {
        Expr::Literal(lit) => match lit {
            Literal::Integer(_span, int) => Term::Number(int.clone().into()),
            Literal::Float(_span, float) => Term::Number((*float).into()),
            Literal::Atom(atom) => Term::Atom(atom.name),

            lit => return Err(EvalError::InvalidConstExpression { span: lit.span() }),
        },

        Expr::Nil(_) => Term::Nil,
        Expr::Cons(cons) => {
            let head = eval_expr(&cons.head, resolve_record_index)?;
            let tail = eval_expr(&cons.tail, resolve_record_index)?;
            Term::Cons(Box::new(head), Box::new(tail))
        }

        Expr::Tuple(tup) => Term::Tuple(
            tup.elements
                .iter()
                .map(|e| eval_expr(e, resolve_record_index))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Map(_) => unimplemented!(),
        Expr::Binary(_) => unimplemented!(),
        Expr::Record(_) => unimplemented!(),
        Expr::RecordIndex(rec_idx) if resolve_record_index.is_some() => {
            match resolve_record_index.unwrap()(rec_idx.name, rec_idx.field) {
                Ok(index) => Term::Number(index.into()),
                Err(ResolveRecordIndexError::NoRecord) => {
                    Err(EvalError::NoRecord { span: rec_idx.span })?
                }
                Err(ResolveRecordIndexError::NoField) => {
                    Err(EvalError::NoRecordField { span: rec_idx.span })?
                }
            }
        }

        Expr::BinaryExpr(bin_expr) => {
            use BinaryOp as B;

            let lhs = eval_expr(&bin_expr.lhs, resolve_record_index)?;
            let rhs = eval_expr(&bin_expr.rhs, resolve_record_index)?;

            match (bin_expr.op, lhs, rhs) {
                (B::Add, Term::Number(l), Term::Number(r)) => (&l + &r)
                    .map_err(|source| EvalError::FloatError { span, source })?
                    .into(),
                (B::Sub, Term::Number(l), Term::Number(r)) => (&l - &r)
                    .map_err(|source| EvalError::FloatError { span, source })?
                    .into(),
                (B::Multiply, Term::Number(l), Term::Number(r)) => (&l * &r)
                    .map_err(|source| EvalError::FloatError { span, source })?
                    .into(),
                (B::Divide, Term::Number(l), Term::Number(r)) => {
                    if r.is_zero() {
                        Err(EvalError::DivisionByZero { span })?
                    }
                    (&l / &r)
                        .map_err(|source| EvalError::FloatError { span, source })?
                        .into()
                }

                (B::Div, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    if r.is_zero() {
                        Err(EvalError::DivisionByZero { span })?
                    }
                    Number::Integer((l / &r).unwrap()).into()
                }
                (B::Div, _, _) => Err(EvalError::InvalidDivOperand { span })?,

                (B::Bor, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    Number::Integer(l | &r).into()
                }
                (B::Band, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    Number::Integer(l & &r).into()
                }
                (B::Bxor, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    Number::Integer(l ^ &r).into()
                }
                (B::Bsl, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    let shift = r.to_u32().ok_or(EvalError::TooLargeShift { span })?;
                    Number::Integer(l << shift).into()
                }
                (B::Bsr, Term::Number(Number::Integer(l)), Term::Number(Number::Integer(r))) => {
                    let shift = r.to_u32().ok_or(EvalError::TooLargeShift { span })?;
                    Number::Integer(l >> shift).into()
                }
                (B::Bor | B::Band | B::Bxor | B::Bsl | B::Bsr, _, _) => {
                    Err(EvalError::InvalidBitwiseOperand { span })?
                }

                (B::Lt, l, r) => (l < r).into(),
                (B::Lte, l, r) => (l <= r).into(),
                (B::Gt, l, r) => (l > r).into(),
                (B::Gte, l, r) => (l >= r).into(),

                (B::Equal, l, r) => l.equals(&r, false).into(),
                (B::NotEqual, l, r) => (!l.equals(&r, false)).into(),
                (B::StrictEqual, l, r) => l.equals(&r, true).into(),
                (B::StrictNotEqual, l, r) => (!l.equals(&r, true)).into(),

                (B::Append, _, _) => unimplemented!(),
                (B::Remove, _, _) => unimplemented!(),

                _ => Err(EvalError::InvalidConstExpression { span })?,
            }
        }

        Expr::UnaryExpr(un_expr) => {
            let operand = eval_expr(&un_expr.operand, resolve_record_index)?;

            match (un_expr.op, operand) {
                (UnaryOp::Plus, Term::Number(o)) => o.plus().into(),
                (UnaryOp::Minus, Term::Number(o)) => (-o).into(),

                (UnaryOp::Bnot, Term::Number(Number::Integer(i))) => Number::Integer(!&i).into(),
                (UnaryOp::Bnot, _) => Err(EvalError::InvalidBitwiseOperand { span })?,

                (UnaryOp::Not, Term::Atom(sym)) if sym == symbols::True => false.into(),
                (UnaryOp::Not, Term::Atom(sym)) if sym == symbols::False => true.into(),

                //(UnaryOp::Not, Term::Atom)
                _ => Err(EvalError::InvalidConstExpression { span })?,
            }
        }

        _ => Err(EvalError::InvalidConstExpression { span })?,
    };
    Ok(res)
}
