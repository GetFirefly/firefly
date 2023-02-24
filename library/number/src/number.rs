use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::{BigInt, DivisionError, Float, FloatError, Int};

#[derive(Debug, Clone, Hash)]
#[repr(u8)]
pub enum Number {
    Integer(Int),
    Float(Float),
}
impl Number {
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Integer(int) => int.is_zero(),
            Self::Float(float) => float.is_zero(),
        }
    }

    pub fn abs(self) -> Self {
        match self {
            Self::Integer(int) => int.abs().into(),
            Self::Float(float) => float.abs().into(),
        }
    }

    pub fn equals(&self, rhs: &Self, exact: bool) -> bool {
        match (self, rhs) {
            (Self::Integer(l), Self::Integer(r)) => l == r,
            (Self::Float(l), Self::Float(r)) => l == r,

            (Self::Integer(_l), Self::Float(_r)) if exact => false,
            (Self::Integer(l), Self::Float(r)) => {
                if r.is_precise() {
                    l.to_float() == r.inner()
                } else {
                    l == &r.to_integer()
                }
            }
            (Self::Float(_), Self::Integer(_)) => rhs.equals(self, exact),
        }
    }

    pub fn to_efloat(&self) -> Result<Float, FloatError> {
        match self {
            Self::Integer(integer) => integer.to_efloat(),
            Self::Float(float) => Ok(*float),
        }
    }
}
impl Eq for Number {}
impl PartialEq for Number {
    fn eq(&self, rhs: &Self) -> bool {
        self.equals(rhs, false)
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Integer(l), Self::Integer(r)) => l.cmp(r),
            (Self::Float(l), Self::Float(r)) => l.partial_cmp(r).unwrap(),
            (Self::Integer(l), Self::Float(r)) => {
                if r.is_precise() {
                    l.to_float().partial_cmp(&r.inner()).unwrap()
                } else {
                    l.cmp(&r.to_integer())
                }
            }
            (Self::Float(l), Self::Integer(r)) => {
                if l.is_precise() {
                    l.inner().partial_cmp(&r.to_float()).unwrap()
                } else {
                    l.to_integer().cmp(r)
                }
            }
        }
    }
}
impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<Int> for Number {
    fn from(int: Int) -> Self {
        Self::Integer(int)
    }
}
impl From<BigInt> for Number {
    fn from(i: BigInt) -> Self {
        Self::Integer(i.into())
    }
}
impl From<i64> for Number {
    fn from(i: i64) -> Self {
        Self::Integer(i.into())
    }
}
impl From<usize> for Number {
    fn from(i: usize) -> Self {
        Self::Integer(i.into())
    }
}
impl From<Float> for Number {
    fn from(n: Float) -> Self {
        Self::Float(n)
    }
}
impl From<f64> for Number {
    fn from(n: f64) -> Self {
        Self::Float(n.into())
    }
}

impl Add<Number> for Number {
    type Output = Result<Number, FloatError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Integer(l), Self::Integer(r)) => Ok(Self::Integer(l + r)),
            (Self::Integer(l), Self::Float(r)) => Ok(Self::Float((l + r)?)),
            (Self::Float(l), Self::Integer(r)) => Ok(Self::Float((l + r)?)),
            (Self::Float(l), Self::Float(r)) => Ok(Self::Float((l + r)?)),
        }
    }
}
impl Add<&Number> for Number {
    type Output = Result<Number, FloatError>;

    fn add(self, rhs: &Self) -> Self::Output {
        self.add(rhs.clone())
    }
}
impl Sub<Number> for Number {
    type Output = Result<Number, FloatError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Integer(l), Self::Integer(r)) => Ok(Self::Integer(l - r)),
            (Self::Integer(l), Self::Float(r)) => Ok(Self::Float((l - r)?)),
            (Self::Float(l), Self::Integer(r)) => Ok(Self::Float((l - r)?)),
            (Self::Float(l), Self::Float(r)) => Ok(Self::Float((l - r)?)),
        }
    }
}
impl Sub<&Number> for Number {
    type Output = Result<Number, FloatError>;

    fn sub(self, rhs: &Self) -> Self::Output {
        self.sub(rhs.clone())
    }
}
impl Mul<Number> for Number {
    type Output = Result<Number, FloatError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Integer(l), Self::Integer(r)) => Ok(Self::Integer(l * r)),
            (Self::Integer(l), Self::Float(r)) => Ok(Self::Float((l * r)?)),
            (Self::Float(l), Self::Integer(r)) => Ok(Self::Float((l * r)?)),
            (Self::Float(l), Self::Float(r)) => Ok(Self::Float((l * r)?)),
        }
    }
}
impl Mul<&Number> for Number {
    type Output = Result<Number, FloatError>;

    fn mul(self, rhs: &Self) -> Self::Output {
        self.mul(rhs.clone())
    }
}
impl Div<Number> for Number {
    type Output = Result<Number, DivisionError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Integer(l), Self::Integer(r)) => Ok(Self::Integer((l / r)?)),
            (Self::Integer(l), Self::Float(r)) => Ok(Self::Float((l / r)?)),
            (Self::Float(l), Self::Integer(r)) => Ok(Self::Float((l / r)?)),
            (Self::Float(l), Self::Float(r)) => Ok(Self::Float((l / r)?)),
        }
    }
}
impl Div<&Number> for Number {
    type Output = Result<Number, DivisionError>;

    fn div(self, rhs: &Self) -> Self::Output {
        self.div(rhs.clone())
    }
}
impl Neg for Number {
    type Output = Number;
    fn neg(self) -> Self {
        match self {
            Self::Integer(int) => Self::Integer(-int),
            Self::Float(float) => Self::Float(-float),
        }
    }
}
