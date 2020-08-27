use core::cmp::Ordering;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display};
use core::ops::*;

use num_bigint::BigInt;

use crate::erts::term::arch::{MAX_SMALLINT_VALUE, MIN_SMALLINT_VALUE};
use crate::erts::term::prelude::*;

use super::*;

/// A small type, slightly less than 64/32-bit wide, as 4 bits are used for tags
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct SmallInteger(pub(in crate::erts::term) isize);
impl SmallInteger {
    pub const MIN_VALUE: isize = MIN_SMALLINT_VALUE;
    pub const MAX_VALUE: isize = MAX_SMALLINT_VALUE;

    /// Create new `SmallInteger` from an `isize` value, returning `Err`
    /// if the value is out of range
    #[inline]
    pub fn new(i: isize) -> Result<Self, TryFromIntError> {
        if i > Self::MAX_VALUE || i < Self::MIN_VALUE {
            return Err(TryFromIntError);
        }
        Ok(Self(i))
    }

    /// Same as `new`, but panics at runtime if the value is out of range
    #[inline]
    pub unsafe fn new_unchecked(i: isize) -> Self {
        assert!(
            i <= Self::MAX_VALUE,
            "invalid small integer value ({}), larger than MAX_VALUE ({})",
            i,
            Self::MAX_VALUE
        );
        assert!(
            i >= Self::MIN_VALUE,
            "invalid small integer value ({}), less than MIN_VALUE ({})",
            i,
            Self::MIN_VALUE
        );
        Self(i)
    }

    /// Returns the number of one bits in the underlying byte representation
    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }

    /// Returns the underlying byte representation, in little-endian order
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl From<u8> for SmallInteger {
    fn from(n: u8) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
impl From<u16> for SmallInteger {
    fn from(n: u16) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
#[cfg(target_pointer_width = "64")]
impl From<u32> for SmallInteger {
    fn from(n: u32) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
#[cfg(target_pointer_width = "32")]
impl TryFrom<u32> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: u32) -> Result<Self, Self::Error> {
        Self::new(n as isize)
    }
}
impl TryFrom<u64> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: u64) -> Result<Self, Self::Error> {
        match n.try_into() {
            Err(_) => Err(TryFromIntError),
            Ok(val) => Self::new(val),
        }
    }
}
impl TryFrom<usize> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: usize) -> Result<Self, Self::Error> {
        match n.try_into() {
            Err(_) => Err(TryFromIntError),
            Ok(val) => Self::new(val),
        }
    }
}
impl From<i8> for SmallInteger {
    fn from(n: i8) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
impl From<i16> for SmallInteger {
    fn from(n: i16) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
#[cfg(target_pointer_width = "64")]
impl From<i32> for SmallInteger {
    fn from(n: i32) -> Self {
        unsafe { Self::new_unchecked(n as isize) }
    }
}
impl TryFrom<i32> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: i32) -> Result<Self, Self::Error> {
        Self::new(n as isize)
    }
}
impl TryFrom<i64> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: i64) -> Result<Self, Self::Error> {
        Self::new(n as isize)
    }
}
impl TryFrom<isize> for SmallInteger {
    type Error = TryFromIntError;

    fn try_from(n: isize) -> Result<Self, Self::Error> {
        Self::new(n)
    }
}
impl Into<isize> for SmallInteger {
    fn into(self) -> isize {
        self.0
    }
}
#[cfg(target_pointer_width = "32")]
impl Into<i32> for SmallInteger {
    fn into(self) -> i32 {
        self.0 as i32
    }
}
#[cfg(target_pointer_width = "64")]
impl TryInto<i32> for SmallInteger {
    type Error = TryFromIntError;

    fn try_into(self) -> Result<i32, Self::Error> {
        self.0.try_into().map_err(|_| TryFromIntError)
    }
}
impl TryInto<u32> for SmallInteger {
    type Error = TryFromIntError;

    fn try_into(self) -> Result<u32, Self::Error> {
        self.0.try_into().map_err(|_| TryFromIntError)
    }
}
impl Into<i64> for SmallInteger {
    fn into(self) -> i64 {
        self.0 as i64
    }
}
impl Into<f64> for SmallInteger {
    fn into(self) -> f64 {
        self.0 as f64
    }
}
impl Into<BigInt> for SmallInteger {
    fn into(self) -> BigInt {
        BigInt::from(self.0)
    }
}
impl TryInto<u8> for SmallInteger {
    type Error = core::num::TryFromIntError;

    fn try_into(self) -> Result<u8, Self::Error> {
        self.0.try_into()
    }
}
impl TryInto<u64> for SmallInteger {
    type Error = core::num::TryFromIntError;

    fn try_into(self) -> Result<u64, Self::Error> {
        self.0.try_into()
    }
}
impl TryInto<usize> for SmallInteger {
    type Error = core::num::TryFromIntError;

    fn try_into(self) -> Result<usize, Self::Error> {
        self.0.try_into()
    }
}
impl Debug for SmallInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("SmallInteger").field(&self.0).finish()
    }
}
impl Display for SmallInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Binary for SmallInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:b}", self.0)
    }
}

macro_rules! smallint_binop_trait_impl {
    ($trait:ty, $fn:ident, $checked:ident) => {
        impl $trait for SmallInteger {
            type Output = Integer;

            #[inline]
            fn $fn(self, rhs: SmallInteger) -> Self::Output {
                match (self.0).$checked(rhs.0) {
                    None => {
                        let lhs: BigInt = self.into();
                        let rhs: BigInt = rhs.into();
                        Integer::Big(BigInteger::new(lhs.$fn(rhs)))
                    }
                    Some(val) if val > Self::MAX_VALUE => Integer::Big(BigInteger::new(val.into())),
                    Some(val) if val < Self::MIN_VALUE => Integer::Big(BigInteger::new(val.into())),
                    Some(val) => Integer::Small(Self(val)),
                }
            }
        }
    };
}
macro_rules! smallint_unaryop_trait_impl {
    ($trait:ty, $fun:ident, $checked:ident) => {
        impl $trait for SmallInteger {
            type Output = Integer;
            #[inline]
            fn $fun(self) -> Self::Output {
                match (self.0).$checked() {
                    None => {
                        let this: BigInt = self.into();
                        Integer::Big(BigInteger::new(this.$fun()))
                    }
                    Some(val) if val > Self::MAX_VALUE => Integer::Big(BigInteger::new(val.into())),
                    Some(val) if val < Self::MIN_VALUE => Integer::Big(BigInteger::new(val.into())),
                    Some(val) => Integer::Small(Self(val)),
                }
            }
        }
    };
}

smallint_binop_trait_impl!(Add, add, checked_add);
smallint_binop_trait_impl!(Sub, sub, checked_sub);
smallint_binop_trait_impl!(Mul, mul, checked_mul);
smallint_binop_trait_impl!(Div, div, checked_div);
smallint_binop_trait_impl!(Rem, rem, checked_rem);
smallint_unaryop_trait_impl!(Neg, neg, checked_neg);

impl Not for SmallInteger {
    type Output = Integer;

    #[inline]
    fn not(self) -> Self::Output {
        // Take the bitwise complement and mask the high bits (the max
        // value has the same bit representation as the desired mask)
        let complement = !self.0 & Self::MAX_VALUE;
        if complement > Self::MAX_VALUE || complement < Self::MIN_VALUE {
            return Integer::Big(BigInteger::new(complement.into()));
        }
        Integer::Small(unsafe { SmallInteger::new_unchecked(complement) })
    }
}
impl Shl<usize> for SmallInteger {
    type Output = Integer;

    #[inline]
    fn shl(self, rhs: usize) -> Self::Output {
        match rhs.try_into() {
            Err(_) => {
                let lhs: BigInt = self.into();
                Integer::Big(BigInteger::new(lhs.shl(rhs)))
            }
            Ok(shift) => match (self.0).checked_shl(shift) {
                None => {
                    let lhs: BigInt = self.into();
                    Integer::Big(BigInteger::new(lhs.shl(shift as usize)))
                }
                Some(val) if val > Self::MAX_VALUE => Integer::Big(BigInteger::new(val.into())),
                Some(val) if val < Self::MIN_VALUE => Integer::Big(BigInteger::new(val.into())),
                Some(val) => Integer::Small(Self(val)),
            },
        }
    }
}
impl Shr<usize> for SmallInteger {
    type Output = Integer;

    #[inline]
    fn shr(self, rhs: usize) -> Self::Output {
        match rhs.try_into() {
            Err(_) => {
                let lhs: BigInt = self.into();
                Integer::Big(BigInteger::new(lhs.shr(rhs)))
            }
            Ok(shift) => match (self.0).checked_shr(shift) {
                None => {
                    let lhs: BigInt = self.into();
                    Integer::Big(BigInteger::new(lhs.shl(shift as usize)))
                }
                Some(val) if val > Self::MAX_VALUE => Integer::Big(BigInteger::new(val.into())),
                Some(val) if val < Self::MIN_VALUE => Integer::Big(BigInteger::new(val.into())),
                Some(val) => Integer::Small(Self(val)),
            },
        }
    }
}

macro_rules! smallint_bitop_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for SmallInteger {
            type Output = Integer;
            #[inline]
            fn $fun(self, rhs: SmallInteger) -> Self::Output {
                Integer::from(unsafe { Self::new_unchecked((self.0).$fun(rhs.0)) })
            }
        }
    };
}

smallint_bitop_trait_impl!(BitAnd, bitand);
smallint_bitop_trait_impl!(BitOr, bitor);
smallint_bitop_trait_impl!(BitXor, bitxor);

impl PartialEq<Float> for SmallInteger {
    #[inline]
    fn eq(&self, other: &Float) -> bool {
        (self.0 as f64).eq(&other.value())
    }
}
impl PartialEq<BigInteger> for SmallInteger {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        other.value.eq(&BigInt::from(self.0 as i64))
    }
}
impl PartialEq<usize> for SmallInteger {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        if self.0 < 0 {
            return false;
        }
        (self.0 as usize).eq(other)
    }
}
impl PartialEq<isize> for SmallInteger {
    #[inline]
    fn eq(&self, other: &isize) -> bool {
        self.0.eq(other)
    }
}
impl<T> PartialEq<Boxed<T>> for SmallInteger
where
    T: PartialEq<SmallInteger>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl PartialOrd<Float> for SmallInteger {
    #[inline]
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        (self.0 as f64).partial_cmp(&other.value())
    }
}
impl PartialOrd<BigInteger> for SmallInteger {
    #[inline]
    fn partial_cmp(&self, other: &BigInteger) -> Option<Ordering> {
        Some(BigInt::from(self.0 as i64).cmp(&other.value))
    }
}
impl<T> PartialOrd<Boxed<T>> for SmallInteger
where
    T: PartialOrd<SmallInteger>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl TryFrom<TypedTerm> for SmallInteger {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::SmallInteger(small_integer) => Ok(small_integer),
            _ => Err(TypeError),
        }
    }
}
