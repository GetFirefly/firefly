use core::any::TypeId;
use core::fmt::{self, Display};
use core::ops::Deref;

use num_bigint::{BigInt, Sign};
use num_traits::ToPrimitive;

use super::Float;

/// BigIntegers are arbitrary-width integers whose size is too large to fit in
/// an immediate/SmallInteger value.
#[derive(Clone, Hash)]
#[repr(transparent)]
pub struct BigInteger(BigInt);
impl BigInteger {
    pub const TYPE_ID: TypeId = TypeId::of::<BigInteger>();

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }

    #[inline]
    pub fn as_float(&self) -> Option<Float> {
        self.0.to_f64().map(|f| f.into())
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        self.0.sign() == Sign::Minus
    }

    #[inline]
    pub fn is_positive(&self) -> bool {
        !self.is_negative()
    }
}
impl Deref for BigInteger {
    type Target = BigInt;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<i64> for BigInteger {
    fn from(value: i64) -> Self {
        Self(value.into())
    }
}
impl From<usize> for BigInteger {
    fn from(value: usize) -> Self {
        Self(value.into())
    }
}
impl From<BigInt> for BigInteger {
    #[inline]
    fn from(value: BigInt) -> Self {
        Self(value)
    }
}
impl TryInto<usize> for &BigInteger {
    type Error = ();

    #[inline]
    fn try_into(self) -> Result<usize, Self::Error> {
        let i = self.as_i64().ok_or(())?;
        i.try_into().map_err(|_| ())
    }
}
impl Eq for BigInteger {}
impl PartialEq for BigInteger {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl PartialEq<i64> for BigInteger {
    fn eq(&self, y: &i64) -> bool {
        match self.as_i64() {
            Some(x) => x.eq(y),
            None => false,
        }
    }
}
impl PartialEq<Float> for BigInteger {
    fn eq(&self, y: &Float) -> bool {
        match self.as_float() {
            Some(x) => y.eq(&x),
            None => false,
        }
    }
}
impl PartialOrd for BigInteger {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd<i64> for BigInteger {
    fn partial_cmp(&self, other: &i64) -> Option<core::cmp::Ordering> {
        let i = self.as_i64()?;
        Some(i.cmp(other))
    }
}
impl PartialOrd<Float> for BigInteger {
    fn partial_cmp(&self, other: &Float) -> Option<core::cmp::Ordering> {
        use core::cmp::Ordering;

        match other.as_f64() {
            x if x.is_infinite() => {
                if x.is_sign_negative() {
                    Some(Ordering::Less)
                } else {
                    Some(Ordering::Greater)
                }
            }
            _ => {
                let too_large = if self.is_negative() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
                let Some(x) = self.as_i64() else { return Some(too_large); };
                Some(other.partial_cmp(&x)?.reverse())
            }
        }
    }
}
impl Ord for BigInteger {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
impl fmt::Debug for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Display for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl core::ops::Add for &BigInteger {
    type Output = BigInteger;

    fn add(self, rhs: &BigInteger) -> Self::Output {
        BigInteger(self.deref().add(&rhs.0))
    }
}
impl core::ops::Add<i64> for &BigInteger {
    type Output = BigInteger;

    fn add(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().add(rhs))
    }
}
impl core::ops::Sub for &BigInteger {
    type Output = BigInteger;

    fn sub(self, rhs: &BigInteger) -> Self::Output {
        BigInteger(self.deref().sub(&rhs.0))
    }
}
impl core::ops::Sub<i64> for &BigInteger {
    type Output = BigInteger;

    fn sub(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().sub(rhs))
    }
}
impl core::ops::Div for &BigInteger {
    type Output = BigInteger;

    fn div(self, rhs: &BigInteger) -> Self::Output {
        BigInteger(self.deref().div(&rhs.0))
    }
}
impl core::ops::Div<i64> for &BigInteger {
    type Output = BigInteger;

    fn div(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().div(rhs))
    }
}
impl core::ops::Rem for &BigInteger {
    type Output = BigInteger;

    fn rem(self, rhs: &BigInteger) -> Self::Output {
        BigInteger(self.deref().rem(&rhs.0))
    }
}
impl core::ops::Rem<i64> for &BigInteger {
    type Output = BigInteger;

    fn rem(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().rem(rhs))
    }
}
impl core::ops::Mul for &BigInteger {
    type Output = BigInteger;

    fn mul(self, rhs: &BigInteger) -> Self::Output {
        BigInteger(self.deref().mul(&rhs.0))
    }
}
impl core::ops::Mul<i64> for &BigInteger {
    type Output = BigInteger;

    fn mul(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().mul(rhs))
    }
}
impl core::ops::Shl<i64> for &BigInteger {
    type Output = BigInteger;

    fn shl(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().shl(rhs))
    }
}
impl core::ops::Shr<i64> for &BigInteger {
    type Output = BigInteger;

    fn shr(self, rhs: i64) -> Self::Output {
        BigInteger(self.deref().shr(rhs))
    }
}
