use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem;
use core::num::FpCategory;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use core::str::FromStr;

pub use half::f16;
use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;

use crate::{DivisionError, Int};

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

impl fmt::Display for FloatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FloatError::Nan => write!(f, "NaN"),
            FloatError::Infinite => write!(f, "Inf"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum ParseFloatError {
    ParseFailed,
    Invalid(FloatError),
}
impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::ParseFailed => write!(f, "could not parse string as float"),
            Self::Invalid(err) => write!(f, "invalid float: {}", &err),
        }
    }
}

/// This is a wrapper around an f64 value that ensures the value is a valid Erlang float, i.e. it
/// cannot be +/- infinity.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Float(f64);
impl Float {
    const I64_UPPER_BOUNDARY: f64 = (1i64 << f64::MANTISSA_DIGITS) as f64;
    const I64_LOWER_BOUNDARY: f64 = (-1i64 << f64::MANTISSA_DIGITS) as f64;

    pub fn new(float: f64) -> Result<Float, FloatError> {
        FloatError::from_category(float.classify())?;
        Ok(Float(float))
    }

    /// Obtain this floating-pointer value as a raw 64-bit value
    #[inline(always)]
    pub fn raw(&self) -> u64 {
        unsafe { mem::transmute(self.0) }
    }

    /// Get this float as a raw f64 value
    #[inline(always)]
    pub fn inner(&self) -> f64 {
        self.0
    }

    /// Return the absolute value of this float
    pub fn abs(&self) -> Float {
        if self.0 < 0.0 {
            Float(self.0 * -1.0)
        } else {
            *self
        }
    }

    /// Returns true if this float is zero
    pub fn is_zero(&self) -> bool {
        self.0.classify() == FpCategory::Zero
    }

    /// Convers this Float to an Int value
    pub fn to_integer(&self) -> Int {
        Int::new(self.0 as i64)
    }

    /// Returns whether this float is more precise than an integer.
    /// If the float is precise, the other integer this is being compared to will
    /// be converted to a float.
    /// If the float is not precise, it will be converted to the integer.
    pub fn is_precise(&self) -> bool {
        self.0 >= Self::I64_LOWER_BOUNDARY && self.0 <= Self::I64_UPPER_BOUNDARY
    }

    /// Returns true if this float is a finite value
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
}
impl FromStr for Float {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.parse::<f64>() {
            Ok(f) => Self::new(f).map_err(ParseFloatError::Invalid),
            Err(_) => Err(ParseFloatError::ParseFailed),
        }
    }
}
impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}
impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Ord for Float {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl From<f64> for Float {
    #[inline(always)]
    fn from(f: f64) -> Self {
        Self(f)
    }
}
impl From<f32> for Float {
    #[inline(always)]
    fn from(f: f32) -> Self {
        Self(f as f64)
    }
}
impl From<i64> for Float {
    #[inline]
    fn from(i: i64) -> Self {
        Self(ToPrimitive::to_f64(&i).unwrap())
    }
}
impl From<f16> for Float {
    #[inline(always)]
    fn from(f: f16) -> Self {
        Self(f.to_f64())
    }
}
impl Into<f64> for Float {
    #[inline(always)]
    fn into(self) -> f64 {
        self.0
    }
}
impl Into<f32> for Float {
    #[inline(always)]
    fn into(self) -> f32 {
        self.0 as f32
    }
}
impl Into<f16> for Float {
    #[inline]
    fn into(self) -> f16 {
        f16::from_f64(self.0)
    }
}
impl ToPrimitive for Float {
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(self.0)
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }

    fn to_i64(&self) -> Option<i64> {
        ToPrimitive::to_i64(&self.0)
    }

    fn to_u64(&self) -> Option<u64> {
        ToPrimitive::to_u64(&self.0)
    }
}

impl Hash for Float {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}
impl Eq for Float {}
impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        if !self.is_finite() || !other.is_finite() {
            return false;
        }
        self.0.eq(&other.0)
    }
}
impl PartialEq<i64> for Float {
    fn eq(&self, y: &i64) -> bool {
        match self.0 {
            x if x.is_infinite() => false,
            x if x >= Self::I64_UPPER_BOUNDARY || x <= Self::I64_LOWER_BOUNDARY => {
                // We're out of the range where f64 is more precise than an i64,
                // so cast the float to integer and comapre.
                //
                // # Safety
                //
                // We've guarded against infinite values, the float cannot be NaN
                // due to our encoding scheme, and we know the value can be represented
                // in i64, so this is guaranteed safe.
                let x: i64 = unsafe { x.to_int_unchecked() };
                x.eq(y)
            }
            x => x.eq(&(*y as f64)),
        }
    }
}
impl PartialEq<BigInt> for Float {
    fn eq(&self, y: &BigInt) -> bool {
        let Some(y) = y.to_i64() else { return false; };
        self.eq(&y)
    }
}
impl PartialEq<Int> for Float {
    fn eq(&self, y: &Int) -> bool {
        match y {
            Int::Small(i) => self.eq(i),
            Int::Big(i) => self.eq(i),
        }
    }
}
impl PartialOrd<i64> for Float {
    fn partial_cmp(&self, y: &i64) -> Option<Ordering> {
        match self.0 {
            x if x.is_infinite() => {
                if x.is_sign_negative() {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            x if x >= Self::I64_UPPER_BOUNDARY || x <= Self::I64_LOWER_BOUNDARY => {
                // We're out of the range where f64 is more precise than an i64,
                // so cast the float to integer and comapre.
                //
                // # Safety
                //
                // We've guarded against infinite values, the float cannot be NaN
                // due to our encoding scheme, and we know the value can be represented
                // in i64, so this is guaranteed safe.
                let x: i64 = unsafe { x.to_int_unchecked() };
                Some(x.cmp(y))
            }
            x => x.partial_cmp(&(*y as f64)),
        }
    }
}
impl PartialOrd<BigInt> for Float {
    fn partial_cmp(&self, y: &BigInt) -> Option<Ordering> {
        match self.0 {
            x if x.is_infinite() => {
                if x.is_sign_negative() {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            _ => {
                let too_large = if y.sign() == Sign::Minus {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
                let Some(y) = y.to_i64() else { return Some(too_large); };
                self.partial_cmp(&y)
            }
        }
    }
}
impl PartialOrd<Int> for Float {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        match other {
            Int::Small(i) => self.partial_cmp(i),
            Int::Big(i) => self.partial_cmp(i),
        }
    }
}

impl Neg for Float {
    type Output = Float;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}
impl Add for Float {
    type Output = Result<Float, FloatError>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.0 + rhs.0)
    }
}
impl Add<i64> for Float {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: i64) -> Self::Output {
        let rhs: Int = rhs.into();
        self + rhs.to_efloat()?
    }
}
impl Add<BigInt> for Float {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: BigInt) -> Self::Output {
        let rhs = Int::Big(rhs);
        self + rhs.to_efloat()?
    }
}
impl Add<Int> for Float {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: Int) -> Self::Output {
        self + rhs.to_efloat()?
    }
}
impl Add<&Int> for Float {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: &Int) -> Self::Output {
        self + rhs.to_efloat()?
    }
}

impl Sub for Float {
    type Output = Result<Float, FloatError>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.0 - rhs.0)
    }
}
impl Sub<i64> for Float {
    type Output = Result<Float, FloatError>;

    fn sub(self, rhs: i64) -> Self::Output {
        let rhs: Int = rhs.into();
        self - rhs.to_efloat()?
    }
}
impl Sub<BigInt> for Float {
    type Output = Result<Float, FloatError>;

    fn sub(self, rhs: BigInt) -> Self::Output {
        let rhs = Int::Big(rhs);
        self - rhs.to_efloat()?
    }
}
impl Sub<Int> for Float {
    type Output = Result<Float, FloatError>;

    fn sub(self, rhs: Int) -> Self::Output {
        self - rhs.to_efloat()?
    }
}
impl Sub<&Int> for Float {
    type Output = Result<Float, FloatError>;
    fn sub(self, rhs: &Int) -> Self::Output {
        self - rhs.to_efloat()?
    }
}

impl Mul for Float {
    type Output = Result<Float, FloatError>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.0 * rhs.0)
    }
}
impl Mul<i64> for Float {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: i64) -> Self::Output {
        let rhs: Int = rhs.into();
        self * rhs.to_efloat()?
    }
}
impl Mul<BigInt> for Float {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: BigInt) -> Self::Output {
        let rhs = Int::Big(rhs);
        self * rhs.to_efloat()?
    }
}
impl Mul<Int> for Float {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: Int) -> Self::Output {
        self * rhs.to_efloat()?
    }
}
impl Mul<&Int> for Float {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: &Int) -> Self::Output {
        self * rhs.to_efloat()?
    }
}

impl Div for Float {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            Err(DivisionError)
        } else {
            Self::new(self.0 / rhs.0).map_err(|_| DivisionError)
        }
    }
}
impl Div<i64> for Float {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: i64) -> Self::Output {
        let rhs: Int = rhs.into();
        self / rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Div<BigInt> for Float {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: BigInt) -> Self::Output {
        let rhs = Int::Big(rhs);
        self / rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Div<Int> for Float {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: Int) -> Self::Output {
        self / rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Div<&Int> for Float {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: &Int) -> Self::Output {
        self / rhs.to_efloat().map_err(|_| DivisionError)?
    }
}

impl Rem for Float {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            Err(DivisionError)
        } else {
            Self::new(self.0 % rhs.0).map_err(|_| DivisionError)
        }
    }
}
impl Rem<i64> for Float {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: i64) -> Self::Output {
        let rhs: Int = rhs.into();
        self % rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Rem<BigInt> for Float {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: BigInt) -> Self::Output {
        let rhs = Int::Big(rhs);
        self % rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Rem<Int> for Float {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: Int) -> Self::Output {
        self % rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
impl Rem<&Int> for Float {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: &Int) -> Self::Output {
        self % rhs.to_efloat().map_err(|_| DivisionError)?
    }
}
