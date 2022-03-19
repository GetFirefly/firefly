use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Float, FloatError, Integer};

#[derive(Debug, Clone, Hash)]
pub enum Number {
    Integer(Integer),
    Float(Float),
}

impl Number {
    pub fn is_zero(&self) -> bool {
        match self {
            Number::Integer(int) => int.is_zero(),
            Number::Float(float) => float.is_zero(),
        }
    }
    pub fn plus(self) -> Self {
        match self {
            Number::Integer(int) => int.plus().into(),
            Number::Float(float) => float.plus().into(),
        }
    }
    pub fn equals(&self, rhs: &Number, exact: bool) -> bool {
        match (self, rhs) {
            (Number::Integer(l), Number::Integer(r)) => l == r,
            (Number::Float(l), Number::Float(r)) => l == r,

            (Number::Integer(_l), Number::Float(_r)) if exact => false,
            (Number::Integer(l), Number::Float(r)) => {
                if r.is_precise() {
                    l.to_float() == r.inner()
                } else {
                    l == &r.to_integer()
                }
            }
            (Number::Float(_), Number::Integer(_)) => rhs.equals(self, exact),
        }
    }

    pub fn to_efloat(&self) -> Result<Float, FloatError> {
        match self {
            Number::Integer(integer) => integer.to_efloat(),
            Number::Float(float) => Ok(*float),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, rhs: &Number) -> bool {
        self.equals(rhs, false)
    }
}
impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Number) -> Ordering {
        match (self, other) {
            (Number::Integer(l), Number::Integer(r)) => l.cmp(r),
            (Number::Float(l), Number::Float(r)) => l.cmp(r),
            (Number::Integer(l), Number::Float(r)) => {
                if r.is_precise() {
                    l.to_float().partial_cmp(&r.inner()).unwrap()
                } else {
                    l.cmp(&r.to_integer())
                }
            }
            (Number::Float(l), Number::Integer(r)) => {
                if l.is_precise() {
                    l.inner().partial_cmp(&r.to_float()).unwrap()
                } else {
                    l.to_integer().cmp(r)
                }
            }
        }
    }
}

impl From<Integer> for Number {
    fn from(int: Integer) -> Number {
        Number::Integer(int)
    }
}
impl From<usize> for Number {
    fn from(int: usize) -> Number {
        Number::Integer(int.into())
    }
}
impl From<Float> for Number {
    fn from(float: Float) -> Number {
        Number::Float(float)
    }
}

impl Neg for Number {
    type Output = Number;
    fn neg(self) -> Number {
        match self {
            Number::Integer(int) => Number::Integer(-int),
            Number::Float(float) => Number::Float(-float),
        }
    }
}

impl Add<&Number> for &Number {
    type Output = Result<Number, FloatError>;
    fn add(self, rhs: &Number) -> Self::Output {
        let res: Number = match (self, rhs) {
            (Number::Integer(l), Number::Integer(r)) => (l.clone() + r).into(),
            (Number::Integer(l), Number::Float(r)) => (l + *r)?.into(),
            (Number::Float(l), Number::Integer(r)) => (*l + r)?.into(),
            (Number::Float(l), Number::Float(r)) => (*l + *r)?.into(),
        };
        Ok(res)
    }
}

impl Sub<&Number> for &Number {
    type Output = Result<Number, FloatError>;
    fn sub(self, rhs: &Number) -> Self::Output {
        let res: Number = match (self, rhs) {
            (Number::Integer(l), Number::Integer(r)) => (l.clone() - r).into(),
            (Number::Integer(l), Number::Float(r)) => (l - *r)?.into(),
            (Number::Float(l), Number::Integer(r)) => (*l - r)?.into(),
            (Number::Float(l), Number::Float(r)) => (*l - *r)?.into(),
        };
        Ok(res)
    }
}

impl Mul<&Number> for &Number {
    type Output = Result<Number, FloatError>;
    fn mul(self, rhs: &Number) -> Self::Output {
        let res: Number = match (self, rhs) {
            (Number::Integer(l), Number::Integer(r)) => (l.clone() * r).into(),
            (Number::Integer(l), Number::Float(r)) => (l * *r)?.into(),
            (Number::Float(l), Number::Integer(r)) => (*l * r)?.into(),
            (Number::Float(l), Number::Float(r)) => (*l * *r)?.into(),
        };
        Ok(res)
    }
}

impl Div<&Number> for &Number {
    type Output = Result<Number, FloatError>;
    fn div(self, rhs: &Number) -> Self::Output {
        let res: Number = match (self, rhs) {
            (Number::Integer(l), Number::Integer(r)) => (l.to_efloat()? + r)?.into(),
            (Number::Integer(l), Number::Float(r)) => (l / *r)?.into(),
            (Number::Float(l), Number::Integer(r)) => (*l / r)?.into(),
            (Number::Float(l), Number::Float(r)) => (*l / *r)?.into(),
        };
        Ok(res)
    }
}
