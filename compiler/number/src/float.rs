use crate::Integer;

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::mem::transmute;
use std::num::FpCategory;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_bigint::BigInt;
use num_traits::FromPrimitive;

#[derive(Debug, Copy, Clone)]
pub struct Float(f64);

impl Float {
    pub fn new(float: f64) -> Result<Float, FloatError> {
        FloatError::from_category(float.classify())?;
        Ok(Float(float))
    }

    #[inline(always)]
    pub fn raw(&self) -> u64 {
        unsafe { transmute(self.0) }
    }

    pub fn inner(&self) -> f64 {
        self.0
    }

    pub fn plus(&self) -> Float {
        if self.0 < 0.0 {
            Float::new(self.0 * -1.0).unwrap()
        } else {
            *self
        }
    }

    pub fn is_zero(&self) -> bool {
        self.0.classify() == FpCategory::Zero
    }

    pub fn to_integer(&self) -> Integer {
        Integer::Big(BigInt::from_f64(self.0).unwrap()).shrink()
    }

    /// Returns whether this float is more precise than an integer.
    /// If the float is precise, the other integer this is being compared to will
    /// be converted to a float.
    /// If the float is not precise, it will be converted to the integer.
    pub fn is_precise(&self) -> bool {
        self.0 > -9007199254740992.0 && self.0 < 9007199254740992.0
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FloatError {
    Nan,
    Infinite,
}
impl FloatError {
    pub fn from_category(category: FpCategory) -> Result<(), Self> {
        match category {
            FpCategory::Nan => Err(FloatError::Nan),
            FpCategory::Infinite => Err(FloatError::Infinite),
            _ => Ok(()),
        }
    }
}

impl std::fmt::Display for FloatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FloatError::Nan => write!(f, "NaN"),
            FloatError::Infinite => write!(f, "Inf"),
        }
    }
}
impl std::error::Error for FloatError {}

macro_rules! impl_op {
    ($trait:ident, $fun:ident) => {
        impl $trait<Float> for Float {
            type Output = Result<Float, FloatError>;
            fn $fun(self, rhs: Float) -> Self::Output {
                Float::new($trait::$fun(self.0, rhs.0))
            }
        }
    };
}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.raw().hash(state)
    }
}

impl PartialEq for Float {
    fn eq(&self, rhs: &Float) -> bool {
        self.raw() == rhs.raw()
    }
}
impl Eq for Float {}

impl PartialOrd for Float {
    fn partial_cmp(&self, rhs: &Float) -> Option<Ordering> {
        self.0.partial_cmp(&rhs.0)
    }
}
impl Ord for Float {
    fn cmp(&self, rhs: &Float) -> Ordering {
        self.partial_cmp(rhs).unwrap()
    }
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);
impl_op!(Rem, rem);

impl Neg for Float {
    type Output = Float;
    fn neg(self) -> Self::Output {
        Float::new(-self.0).unwrap()
    }
}

impl std::fmt::Display for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add<&Integer> for Float {
    type Output = Result<Float, FloatError>;
    fn add(self, rhs: &Integer) -> Self::Output {
        self + rhs.to_efloat()?
    }
}
impl Add<Float> for &Integer {
    type Output = Result<Float, FloatError>;
    fn add(self, rhs: Float) -> Self::Output {
        self.to_efloat()? + rhs
    }
}

impl Sub<&Integer> for Float {
    type Output = Result<Float, FloatError>;
    fn sub(self, rhs: &Integer) -> Self::Output {
        self - rhs.to_efloat()?
    }
}
impl Sub<Float> for &Integer {
    type Output = Result<Float, FloatError>;
    fn sub(self, rhs: Float) -> Self::Output {
        self.to_efloat()? - rhs
    }
}

impl Mul<&Integer> for Float {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: &Integer) -> Self::Output {
        self * rhs.to_efloat()?
    }
}
impl Mul<Float> for &Integer {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: Float) -> Self::Output {
        self.to_efloat()? * rhs
    }
}

impl Div<&Integer> for Float {
    type Output = Result<Float, FloatError>;
    fn div(self, rhs: &Integer) -> Self::Output {
        self / rhs.to_efloat()?
    }
}
impl Div<Float> for &Integer {
    type Output = Result<Float, FloatError>;
    fn div(self, rhs: Float) -> Self::Output {
        self.to_efloat()? / rhs
    }
}

impl Rem<&Integer> for Float {
    type Output = Result<Float, FloatError>;
    fn rem(self, rhs: &Integer) -> Self::Output {
        self % rhs.to_efloat()?
    }
}
impl Rem<Float> for &Integer {
    type Output = Result<Float, FloatError>;
    fn rem(self, rhs: Float) -> Self::Output {
        self.to_efloat()? % rhs
    }
}
