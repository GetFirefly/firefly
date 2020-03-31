use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(not(target_arch = "x86_64"))] {
        mod packed;
        use self::packed as layout;
    } else if #[cfg(target_arch = "x86_64")] {
        mod immediate;
        use self::immediate as layout;
    }
}

// Export the target-specific float representation
pub use layout::Float;

use core::cmp::Ordering;
use core::fmt::{self, Display};
use core::hash::{self, Hash};
use core::ops::*;

use num_bigint::{BigInt, Sign};

use super::prelude::*;

impl Float {
    pub const INTEGRAL_MIN: f64 = -9007199254740992.0;
    pub const INTEGRAL_MAX: f64 = 9007199254740992.0;

    pub fn clamp_inclusive_range(overflowing_range: RangeInclusive<f64>) -> RangeInclusive<f64> {
        Self::clamp_value(overflowing_range.start().clone())
            ..=Self::clamp_value(overflowing_range.end().clone())
    }

    fn clamp_value(overflowing: f64) -> f64 {
        if overflowing == core::f64::NEG_INFINITY {
            core::f64::MIN
        } else if overflowing == core::f64::INFINITY {
            core::f64::MAX
        } else {
            overflowing
        }
    }
}

impl From<SmallInteger> for Float {
    #[inline]
    fn from(n: SmallInteger) -> Self {
        Self::new(n.0 as f64)
    }
}
impl From<f64> for Float {
    #[inline]
    fn from(f: f64) -> Self {
        Self::new(f)
    }
}
impl Into<f64> for Float {
    #[inline]
    fn into(self) -> f64 {
        self.value()
    }
}

impl<T> PartialEq<Boxed<T>> for Float
where
    T: PartialEq<Float>,
{
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}
impl PartialEq<f64> for Float {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.value() == *other
    }
}
impl PartialEq<SmallInteger> for Float {
    #[inline]
    fn eq(&self, other: &SmallInteger) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Equal) => true,
            _ => false,
        }
    }
}
impl PartialEq<BigInteger> for Float {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Equal) => true,
            _ => false,
        }
    }
}

impl<T> PartialOrd<Boxed<T>> for Float
where
    T: PartialOrd<Float>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}
impl PartialOrd<f64> for Float {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.value().partial_cmp(other)
    }
}
impl PartialOrd<SmallInteger> for Float {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        use core::num::FpCategory;

        let value = self.value();
        let is_negative = value.is_sign_negative();
        match value.classify() {
            FpCategory::Nan => None,
            FpCategory::Subnormal => {
                // The float is less precise, so convert to isize
                let f = value as isize;
                Some(f.cmp(&other.0))
            }
            FpCategory::Infinite if is_negative => Some(Ordering::Less),
            FpCategory::Infinite => Some(Ordering::Greater),
            FpCategory::Zero => Some(0.cmp(&other.0)),
            FpCategory::Normal => {
                // Float is higher precision
                let i = other.0 as f64;
                value.partial_cmp(&i)
            }
        }
    }
}
impl PartialOrd<BigInteger> for Float {
    #[inline]
    fn partial_cmp(&self, other: &BigInteger) -> Option<Ordering> {
        use core::num::FpCategory;
        use num_traits::Zero;

        let value = self.value();
        let is_negative = value.is_sign_negative();
        match value.classify() {
            FpCategory::Nan => None,
            FpCategory::Subnormal => {
                // The float is less precise, so convert to isize
                let f = BigInt::from(value as isize);
                Some(f.cmp(&other.value))
            }
            FpCategory::Infinite if is_negative => Some(Ordering::Less),
            FpCategory::Infinite => Some(Ordering::Greater),
            FpCategory::Zero => {
                let f: BigInt = Zero::zero();
                Some(f.cmp(&other.value))
            }
            FpCategory::Normal => {
                use num_traits::ToPrimitive;
                // Float is higher precision, try and convert to it,
                // if we fail, then the bigint is larger in either direction,
                // which we must determine based on its sign
                if let Some(i) = other.value.to_isize() {
                    return value.partial_cmp(&(i as f64));
                }
                if let Sign::Minus = other.value.sign() {
                    return Some(Ordering::Greater);
                }
                Some(Ordering::Less)
            }
        }
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use Debug format so that decimal point is always included so that it is obvious it is a
        // float and not an integer
        write!(f, "{:?}", self.value())
    }
}
impl Hash for Float {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.value().to_bits().hash(state);
    }
}

macro_rules! float_op_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for Float {
            type Output = Float;
            #[inline]
            fn $fun(self, rhs: Float) -> Self::Output {
                Self::new(self.value().$fun(rhs.value()))
            }
        }
    };
}

float_op_trait_impl!(Add, add);
float_op_trait_impl!(Sub, sub);
float_op_trait_impl!(Mul, mul);
float_op_trait_impl!(Div, div);
float_op_trait_impl!(Rem, rem);

impl Neg for Float {
    type Output = Float;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(self.value().neg())
    }
}
