use liblumen_diagnostics::ByteSpan;

use super::{Ident, Name};
use super::{Literal, BinaryOp, UnaryOp, Expr, BinaryElement};

#[derive(Debug, Clone)]
pub enum Pattern {
    Var(Ident),
    Literal(Literal),
    Nil(ByteSpan),
    Cons(ByteSpan, Box<Pattern>, Box<Pattern>),
    Tuple(ByteSpan, Vec<Expr>),
    Binary(ByteSpan, Vec<BinaryElement>),
    Map(ByteSpan, Option<Box<Pattern>>, Vec<MapFieldPattern>),
    Record(ByteSpan, Ident, Vec<RecordFieldPattern>),
    RecordIndex(ByteSpan, Ident, Ident),
    Match(ByteSpan, Box<Pattern>, Box<Pattern>),
    BinaryExpr(ByteSpan, Box<Pattern>, BinaryOp, Box<Pattern>),
    UnaryExpr(ByteSpan, UnaryOp, Box<Pattern>),
}
impl PartialEq for Pattern {
    fn eq(&self, other: &Pattern) -> bool {
        let left = std::mem::discriminant(self);
        let right = std::mem::discriminant(other);
        if left != right {
            return false;
        }

        match (self, other) {
            (&Pattern::Var(ref x), &Pattern::Var(ref y)) => x == y,
            (&Pattern::Literal(ref x), &Pattern::Literal(ref y)) => x == y,
            (&Pattern::Nil(_), &Pattern::Nil(_)) => true,
            (&Pattern::Cons(_, ref x1, ref x2), &Pattern::Cons(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (&Pattern::Tuple(_, ref x), &Pattern::Tuple(_, ref y)) => x == y,
            (&Pattern::Binary(_, ref x), &Pattern::Binary(_, ref y)) => x == y,
            (&Pattern::Map(_, ref x1, ref x2), &Pattern::Map(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (&Pattern::Record(_, ref x1, ref x2), &Pattern::Record(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (
                &Pattern::RecordIndex(_, ref x1, ref x2),
                &Pattern::RecordIndex(_, ref y1, ref y2),
            ) => (x1 == y1) && (x2 == y2),
            (&Pattern::Match(_, ref x1, ref x2), &Pattern::Match(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            (
                &Pattern::BinaryExpr(_, ref x1, ref x2, ref x3),
                &Pattern::BinaryExpr(_, ref y1, ref y2, ref y3),
            ) => (x1 == y1) && (x2 == y2) && (x3 == y3),
            (&Pattern::UnaryExpr(_, ref x1, ref x2), &Pattern::UnaryExpr(_, ref y1, ref y2)) => {
                (x1 == y1) && (x2 == y2)
            }
            _ => false,
        }
    }
}


#[derive(Debug, Clone)]
pub enum MapFieldPattern {
    span: ByteSpan,
    key: Pattern,
    value: Pattern,
}
impl PartialEq for MapFieldPattern {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value
    }
}


#[derive(Debug, Clone)]
pub struct RecordFieldPattern {
    pub span: ByteSpan,
    pub name: Name,
    pub value: Option<Pattern>,
}
impl PartialEq for RecordFieldPattern {
    fn eq(&self, other: &RecordFieldPattern) -> bool {
        (self.name == other.name) && (self.value == other.value)
    }
}
