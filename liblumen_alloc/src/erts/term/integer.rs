mod big;
mod small;

pub use big::*;
pub use small::*;

use num_bigint::{BigInt, Sign};

use core::cmp::Ordering;
use core::convert::{TryInto, TryFrom};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ops::*;

use crate::erts::{AsTerm, Term};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TryFromIntError;
impl Display for TryFromIntError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("attempted to convert to small integer with out of range value")
    }
}
impl From<core::num::TryFromIntError> for TryFromIntError {
    fn from(_: core::num::TryFromIntError) -> Self {
        TryFromIntError
    }
}

macro_rules! unwrap_integer_self {
    ($i:expr => $name:ident => $blk:block) => {
        match $i {
            &Self::Big(ref $name) => $blk,
            &Self::Small(ref $name) => $blk,
        }
    };
}

#[derive(Clone)]
pub enum Arch64Integer {
    Small(u64),
    Big(BigInt),
}
impl From<u64> for Arch64Integer {
    fn from(value: u64) -> Self {
        Self::Small(value)
    }
}
impl From<BigInt> for Arch64Integer {
    fn from(value: BigInt) -> Self {
        Self::Big(value)
    }
}

#[derive(Clone)]
pub enum Arch32Integer {
    Small(u32),
    Big(BigInt),
}
impl From<u32> for Arch32Integer {
    fn from(value: u32) -> Self {
        Self::Small(value)
    }
}
impl TryFrom<u64> for Arch32Integer {
    type Error = TryFromIntError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value.try_into() {
            Err(_) => Err(TryFromIntError),
            Ok(i) => Ok(Self::Small(i))
        }
    }
}
impl From<BigInt> for Arch32Integer {
    fn from(value: BigInt) -> Self {
        Self::Big(value)
    }
}

/// A wrapped type for integers that transparently handles promotion/demotion
#[derive(Clone)]
pub enum Integer {
    Small(SmallInteger),
    Big(BigInteger),
}
impl Integer {
    #[inline]
    pub fn map<S, B>(self, small: S, big: B) -> Self
    where
        S: FnOnce(SmallInteger) -> Integer,
        B: FnOnce(BigInteger) -> BigInteger,
    {
        match self {
            Self::Small(value) => small(value),
            Self::Big(value) => Integer::from(big(value)),
        }
    }

    #[inline]
    pub fn map_pair<S, B>(self, other: Integer, small: S, big: B) -> Self
    where
        S: FnOnce(SmallInteger, SmallInteger) -> Integer,
        B: FnOnce(BigInteger, BigInteger) -> BigInteger,
    {
        match (self, other) {
            (Self::Small(lhs), Self::Small(rhs)) => small(lhs, rhs),
            (Self::Big(lhs), Self::Big(rhs)) => Integer::from(big(lhs, rhs)),
            (Self::Small(lhs), Self::Big(rhs)) => Integer::from(big(lhs.into(), rhs)),
            (Self::Big(lhs), Self::Small(rhs)) => Integer::from(big(lhs, rhs.into())),
        }
    }

    #[inline]
    pub fn map_pair_to<S, B, T>(self, other: Integer, small: S, big: B) -> T
    where
        S: FnOnce(SmallInteger, SmallInteger) -> T,
        B: FnOnce(BigInteger, BigInteger) -> T,
    {
        match (self, other) {
            (Self::Small(lhs), Self::Small(rhs)) => small(lhs, rhs),
            (Self::Big(lhs), Self::Big(rhs)) => big(lhs, rhs),
            (Self::Small(lhs), Self::Big(rhs)) => big(lhs.into(), rhs),
            (Self::Big(lhs), Self::Small(rhs)) => big(lhs, rhs.into()),
        }
    }
}
impl From<BigInt> for Integer {
    #[inline]
    fn from(big_int: BigInt) -> Self {
        let small_min_big_int: BigInt = SmallInteger::MIN_VALUE.into();
        let small_max_big_int: BigInt = SmallInteger::MAX_VALUE.into();

        if (small_min_big_int <= big_int) && (big_int <= small_max_big_int) {
            let (sign, bytes) = big_int.to_bytes_be();
            let small_usize = bytes
                .iter()
                .fold(0_usize, |acc, byte| (acc << 8) | (*byte as usize));

            let small_isize = match sign {
                Sign::Minus => -1 * (small_usize as isize),
                _ => small_usize as isize,
            };

            Integer::Small(SmallInteger(small_isize))
        } else {
            Integer::Big(BigInteger::new(big_int))
        }
    }
}
impl From<BigInteger> for Integer {
    #[inline]
    fn from(i: BigInteger) -> Self {
        Self::Big(i)
    }
}
impl From<SmallInteger> for Integer {
    #[inline]
    fn from(i: SmallInteger) -> Self {
        Self::Small(i)
    }
}
impl From<char> for Integer {
    #[inline]
    fn from(c: char) -> Integer {
        (c as usize).into()
    }
}
impl From<u8> for Integer {
    fn from(n: u8) -> Integer {
        Integer::Small(unsafe { SmallInteger::new_unchecked(n as isize) })
    }
}
impl From<u64> for Integer {
    fn from(n: u64) -> Integer {
        let ni: Result<isize, _> = n.try_into();
        match ni {
            Err(_) => Integer::Big(n.into()),
            Ok(n) if n > SmallInteger::MAX_VALUE => Integer::Big(n.into()),
            Ok(n) => Integer::Small(unsafe { SmallInteger::new_unchecked(n) }),
        }
    }
}
impl From<usize> for Integer {
    #[inline]
    fn from(n: usize) -> Integer {
        let ni: Result<isize, _> = n.try_into();
        match ni {
            Err(_) => Integer::Big(n.into()),
            Ok(n) if n > SmallInteger::MAX_VALUE => Integer::Big(n.into()),
            Ok(n) => Integer::Small(unsafe { SmallInteger::new_unchecked(n) }),
        }
    }
}
impl From<i32> for Integer {
    fn from(n: i32) -> Integer {
        (n as i64).into()
    }
}
impl From<i64> for Integer {
    fn from(n: i64) -> Integer {
        if (SmallInteger::MIN_VALUE as i64) <= n && n <= (SmallInteger::MAX_VALUE as i64) {
            Integer::Small(unsafe { SmallInteger::new_unchecked(n as isize) })
        } else {
            Integer::Big(n.into())
        }
    }
}
impl From<isize> for Integer {
    #[inline]
    fn from(n: isize) -> Integer {
        (n as i64).into()
    }
}

impl PartialEq for Integer {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&Integer::Small(ref lhs), &Integer::Small(ref rhs)) => lhs.eq(rhs),
            (&Integer::Big(ref lhs), &Integer::Big(ref rhs)) => lhs.eq(rhs),
            (&Integer::Small(ref lhs), &Integer::Big(ref rhs)) => lhs.eq(rhs),
            (&Integer::Big(ref lhs), &Integer::Small(ref rhs)) => lhs.eq(rhs),
        }
    }
}
impl PartialEq<SmallInteger> for Integer {
    #[inline]
    fn eq(&self, other: &SmallInteger) -> bool {
        match self {
            &Integer::Small(ref lhs) => lhs.eq(other),
            &Integer::Big(ref lhs) => lhs.eq(other),
        }
    }
}
impl PartialEq<BigInteger> for Integer {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        match self {
            &Integer::Small(ref lhs) => lhs.eq(other),
            &Integer::Big(ref lhs) => lhs.eq(other),
        }
    }
}
impl Eq for Integer {}
impl Ord for Integer {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (&Integer::Small(ref lhs), &Integer::Small(ref rhs)) => lhs.cmp(rhs),
            (&Integer::Big(ref lhs), &Integer::Big(ref rhs)) => lhs.cmp(rhs),
            (&Integer::Small(ref lhs), &Integer::Big(ref rhs)) => lhs.partial_cmp(rhs).unwrap(),
            (&Integer::Big(ref lhs), &Integer::Small(ref rhs)) => lhs.partial_cmp(rhs).unwrap(),
        }
    }
}
impl PartialOrd for Integer {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd<SmallInteger> for Integer {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        match self {
            &Integer::Small(ref lhs) => lhs.partial_cmp(other),
            &Integer::Big(ref lhs) => lhs.partial_cmp(other),
        }
    }
}
impl PartialOrd<BigInteger> for Integer {
    #[inline]
    fn partial_cmp(&self, other: &BigInteger) -> Option<Ordering> {
        match self {
            &Integer::Big(ref lhs) => lhs.partial_cmp(other),
            &Integer::Small(ref lhs) => lhs.partial_cmp(other),
        }
    }
}
impl Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Self::Big(ref value) => f
                .debug_tuple("Integer::Big")
                .field(&format_args!("{:?}", value))
                .finish(),
            &Self::Small(ref value) => f
                .debug_tuple("Integer::Small")
                .field(&format_args!("{:?}", value))
                .finish(),
        }
    }
}
impl Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unwrap_integer_self!(self => value => { write!(f, "{}", value) })
    }
}
impl Hash for Integer {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        unwrap_integer_self!(self => value => { value.hash(state) })
    }
}
unsafe impl AsTerm for Integer {
    unsafe fn as_term(&self) -> Term {
        match self {
            &Self::Small(ref i) => i.as_term(),
            &Self::Big(ref i) => i.as_term(),
        }
    }
}

macro_rules! integer_binop_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for Integer {
            type Output = Integer;
            #[inline]
            fn $fun(self, rhs: Integer) -> Self::Output {
                self.map_pair(rhs, |l, r| l.$fun(r), |l, r| l.$fun(r))
                    .into()
            }
        }
    };
}

macro_rules! integer_unaryop_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for Integer {
            type Output = Integer;
            #[inline]
            fn $fun(self) -> Self::Output {
                self.map(|i| i.$fun(), |i| i.$fun())
            }
        }
    };
}

integer_binop_trait_impl!(Add, add);
integer_binop_trait_impl!(Sub, sub);
integer_binop_trait_impl!(Mul, mul);
integer_binop_trait_impl!(Div, div);
integer_binop_trait_impl!(BitAnd, bitand);
integer_binop_trait_impl!(BitOr, bitor);
integer_binop_trait_impl!(BitXor, bitxor);
integer_binop_trait_impl!(Rem, rem);
integer_unaryop_trait_impl!(Neg, neg);
integer_unaryop_trait_impl!(Not, not);

impl Shl<usize> for Integer {
    type Output = Integer;

    fn shl(self, rhs: usize) -> Self {
        self.map(|n| n.shl(rhs), |n| n.shl(rhs))
    }
}
impl Shr<usize> for Integer {
    type Output = Integer;

    fn shr(self, rhs: usize) -> Self {
        self.map(|n| n.shr(rhs), |n| n.shr(rhs))
    }
}
