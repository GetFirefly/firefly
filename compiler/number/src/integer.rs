use crate::{DivisionError, Float, FloatError};

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use core::str::FromStr;

pub use num_bigint::ToBigInt;
use num_bigint::{BigInt, ParseBigIntError};
pub use num_traits::{FromPrimitive, Signed, ToPrimitive, Zero};

#[derive(Debug, Clone, Hash)]
pub enum Integer {
    Small(i64),
    Big(BigInt),
}

impl ToBigInt for Integer {
    fn to_bigint(&self) -> Option<BigInt> {
        match self {
            Self::Small(int) => Some(BigInt::from(*int)),
            Self::Big(num) => Some(num.clone()),
        }
    }
}

impl Integer {
    // NOTE: See OpaqueTerm in liblumen_rt for the authoritative source of these constants
    const NAN: u64 = unsafe { core::mem::transmute::<f64, u64>(f64::NAN) };
    const QUIET_BIT: u64 = 1 << 51;
    const SIGN_BIT: u64 = 1 << 63;
    const INFINITY: u64 = Self::NAN & !Self::QUIET_BIT;
    const INTEGER_TAG: u64 = Self::INFINITY | Self::SIGN_BIT;
    const NEG: u64 = Self::INTEGER_TAG | Self::QUIET_BIT;
    const MIN_SMALL: i64 = (Self::NEG as i64);
    const MAX_SMALL: i64 = (!Self::NEG as i64);

    pub fn new(i: i64) -> Self {
        if i < Self::MIN_SMALL || i > Self::MAX_SMALL {
            Self::Big(i.into())
        } else {
            Self::Small(i)
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::Small(num) => *num == 0,
            Self::Big(num) => num.is_zero(),
        }
    }

    pub fn abs(&self) -> Self {
        match self {
            Self::Small(num) => Self::Small((*num).abs()),
            Self::Big(num) => Self::Big(num.abs()),
        }
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Self::Small(int) => *int as f64,
            Self::Big(int) => crate::bigint_to_double(int),
        }
    }

    pub fn to_efloat(&self) -> Result<Float, FloatError> {
        Float::new(self.to_float())
    }

    pub fn shrink(self) -> Self {
        match self {
            Self::Small(i) => Self::new(i),
            Self::Big(i) => {
                if let Some(s) = i.to_i64() {
                    Self::new(s)
                } else {
                    Self::Big(i)
                }
            }
        }
    }

    pub fn from_string_radix(string: &str, radix: u32) -> Option<Self> {
        if let Ok(i) = i64::from_str_radix(string, radix) {
            return Some(Self::new(i));
        }
        let bi = BigInt::parse_bytes(string.as_bytes(), radix)?;
        Some(Self::Big(bi))
    }

    pub fn to_arity(&self) -> u8 {
        match self {
            Self::Small(i) => (*i).try_into().unwrap(),
            Self::Big(_) => {
                panic!("invalid arity, expected value within u8 range, but got big integer")
            }
        }
    }

    pub fn to_char(&self) -> Option<char> {
        match self {
            Self::Small(i) => (*i).try_into().ok().and_then(char::from_u32),
            _ => None,
        }
    }

    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Self::Small(i) => (*i).try_into().ok(),
            _ => None,
        }
    }

    /// Determines the fewest bits necessary to express this integer value, not including the sign
    pub fn bits(&self) -> u64 {
        match self {
            Self::Big(i) => i.bits(),
            Self::Small(i) => {
                let i = *i;
                if i >= 0 {
                    (64 - i.leading_zeros()) as u64
                } else {
                    (64 - i.leading_ones()) as u64
                }
            }
        }
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Small(int) => int.fmt(f),
            Self::Big(int) => int.fmt(f),
        }
    }
}

impl FromStr for Integer {
    type Err = ParseBigIntError;
    fn from_str(s: &str) -> Result<Self, ParseBigIntError> {
        match s.parse::<i64>() {
            Ok(i) => Ok(Self::new(i)),
            Err(_) => match s.parse::<BigInt>() {
                Ok(int) => Ok(Self::Big(int)),
                Err(err) => Err(err),
            },
        }
    }
}

impl Eq for Integer {}
impl PartialEq for Integer {
    fn eq(&self, rhs: &Integer) -> bool {
        match (self, rhs) {
            (Self::Small(lhs), Self::Small(rhs)) => lhs.eq(rhs),
            (Self::Small(lhs), Self::Big(rhs)) => {
                if let Some(ref i) = rhs.to_i64() {
                    return lhs.eq(i);
                }
                false
            }
            (Self::Big(lhs), Self::Small(rhs)) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.eq(rhs);
                }
                false
            }
            (Self::Big(lhs), Self::Big(rhs)) => lhs.eq(rhs),
        }
    }
}
impl PartialEq<f64> for Integer {
    fn eq(&self, rhs: &f64) -> bool {
        match self {
            Self::Small(lhs) => (*lhs as f64).eq(rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).eq(rhs),
        }
    }
}
impl PartialEq<Float> for Integer {
    fn eq(&self, rhs: &Float) -> bool {
        match self {
            Self::Small(lhs) => (*lhs as f64) == rhs.inner(),
            Self::Big(lhs) => crate::bigint_to_double(lhs) == rhs.inner(),
        }
    }
}
impl PartialEq<Integer> for f64 {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}
impl PartialEq<char> for Integer {
    fn eq(&self, rhs: &char) -> bool {
        match self {
            Self::Small(lhs) => lhs.eq(&(*rhs as i64)),
            Self::Big(lhs) => {
                let rhs = BigInt::from(*rhs as i64);
                lhs.eq(&rhs)
            }
        }
    }
}
impl PartialEq<Integer> for char {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}
impl PartialEq<i64> for Integer {
    fn eq(&self, rhs: &i64) -> bool {
        match self {
            Self::Small(lhs) => lhs.eq(rhs),
            Self::Big(lhs) => {
                let rhs = BigInt::from(*rhs);
                lhs.eq(&rhs)
            }
        }
    }
}
impl PartialEq<Integer> for i64 {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}

impl Ord for Integer {
    fn cmp(&self, rhs: &Self) -> Ordering {
        match (self, rhs) {
            (Self::Small(lhs), Self::Small(rhs)) => lhs.cmp(rhs),
            (Self::Small(lhs), Self::Big(rhs)) => {
                let lhs = BigInt::from(*lhs);
                lhs.cmp(rhs)
            }
            (Self::Big(lhs), Self::Small(rhs)) => {
                let rhs = BigInt::from(*rhs);
                lhs.cmp(&rhs)
            }
            (Self::Big(lhs), Self::Big(rhs)) => lhs.cmp(rhs),
        }
    }
}
impl PartialOrd for Integer {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl PartialOrd<f64> for Integer {
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        match self {
            Self::Small(lhs) => (*lhs as f64).partial_cmp(rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).partial_cmp(rhs),
        }
    }
}
impl PartialOrd<Float> for Integer {
    fn partial_cmp(&self, rhs: &Float) -> Option<Ordering> {
        let rhs = rhs.inner();
        match self {
            Self::Small(lhs) => (*lhs as f64).partial_cmp(&rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).partial_cmp(&rhs),
        }
    }
}
impl PartialOrd<Integer> for f64 {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl PartialOrd<char> for Integer {
    fn partial_cmp(&self, rhs: &char) -> Option<Ordering> {
        match self {
            Self::Small(lhs) => lhs.partial_cmp(&(*rhs as i64)),
            Self::Big(lhs) => {
                let rhs = BigInt::from(*rhs as i64);
                lhs.partial_cmp(&rhs)
            }
        }
    }
}
impl PartialOrd<Integer> for char {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}
impl PartialOrd<i64> for Integer {
    fn partial_cmp(&self, rhs: &i64) -> Option<Ordering> {
        match self {
            Self::Small(lhs) => lhs.partial_cmp(rhs),
            Self::Big(lhs) => {
                let rhs = BigInt::from(*rhs);
                lhs.partial_cmp(&rhs)
            }
        }
    }
}
impl PartialOrd<Integer> for i64 {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl Shr<u32> for Integer {
    type Output = Integer;
    fn shr(self, num: u32) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(i.checked_shr(num).unwrap_or(0)),
            Self::Big(i) => Self::Big(i >> num).shrink(),
        }
    }
}
impl Shl<u32> for Integer {
    type Output = Integer;
    fn shl(self, num: u32) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_shl(num) {
                None => {
                    let i = BigInt::from(i);
                    Self::Big(i << num)
                }
                Some(i) => Self::new(i),
            },
            Self::Big(i) => Self::Big(i << num),
        }
    }
}
impl Mul<usize> for &Integer {
    type Output = Integer;
    fn mul(self, rhs: usize) -> Self::Output {
        match rhs.try_into() {
            Ok(rhs) => match self {
                Integer::Small(i) => match i.checked_mul(rhs) {
                    None => {
                        let i = BigInt::from(*i);
                        Integer::Big(i * rhs)
                    }
                    Some(i) => Integer::new(i),
                },
                Integer::Big(i) => Integer::Big(i * rhs),
            },
            Err(_) => {
                let lhs = self.to_bigint().unwrap();
                Integer::Big(lhs * BigInt::from(rhs))
            }
        }
    }
}
impl Mul<i64> for Integer {
    type Output = Integer;
    fn mul(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_mul(rhs) {
                None => {
                    let i = BigInt::from(i);
                    Self::Big(i * rhs)
                }
                Some(i) => Self::new(i),
            },
            Self::Big(i) => Self::Big(i * rhs),
        }
    }
}
impl Mul<&BigInt> for Integer {
    type Output = Integer;
    fn mul(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => {
                let lhs = BigInt::from(i);
                Self::Big(lhs * rhs)
            }
            Self::Big(i) => Self::Big(i * rhs),
        }
    }
}
impl Mul<&Integer> for Integer {
    type Output = Integer;
    fn mul(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.mul(*rhs),
            Self::Big(rhs) => self.mul(rhs),
        }
    }
}
impl Mul<Integer> for Integer {
    type Output = Integer;
    fn mul(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.mul(rhs),
            Self::Big(rhs) => self.mul(&rhs),
        }
    }
}
impl Div<i64> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn div(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_div(rhs) {
                None if rhs == 0 => Err(DivisionError),
                None => {
                    let i = BigInt::from(i);
                    Ok(Self::Big(i / rhs))
                }
                Some(i) => Ok(Self::new(i)),
            },
            Self::Big(_) if rhs == 0 => Err(DivisionError),
            Self::Big(i) => Ok(Self::Big(i / rhs)),
        }
    }
}
impl Div<&BigInt> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn div(self, rhs: &BigInt) -> Self::Output {
        if rhs.is_zero() {
            return Err(DivisionError);
        }

        match self {
            Self::Small(i) => {
                let lhs = BigInt::from(i);
                Ok(Self::Big(lhs / rhs))
            }
            Self::Big(i) => Ok(Self::Big(i / rhs)),
        }
    }
}
impl Div<&Integer> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn div(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.div(*rhs),
            Self::Big(rhs) => self.div(rhs),
        }
    }
}
impl Div<Integer> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn div(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.div(rhs),
            Self::Big(rhs) => self.div(&rhs),
        }
    }
}
impl Add<i64> for Integer {
    type Output = Integer;

    fn add(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_add(rhs) {
                None => {
                    let i = BigInt::from(i);
                    Self::Big(i + rhs)
                }
                Some(i) => Self::new(i),
            },
            Self::Big(i) => Self::Big(i + rhs),
        }
    }
}
impl Add<&BigInt> for Integer {
    type Output = Integer;

    fn add(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => {
                let i = BigInt::from(i);
                Self::Big(i + rhs)
            }
            Self::Big(i) => Self::Big(i + rhs),
        }
    }
}
impl Add<&Integer> for Integer {
    type Output = Integer;

    fn add(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.add(*rhs),
            Self::Big(rhs) => self.add(rhs),
        }
    }
}
impl Add<Integer> for Integer {
    type Output = Integer;

    fn add(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.add(rhs),
            Self::Big(rhs) => self.add(&rhs),
        }
    }
}
impl Sub<i64> for Integer {
    type Output = Integer;

    fn sub(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_sub(rhs) {
                None => {
                    let i = BigInt::from(i);
                    Self::Big(i - rhs)
                }
                Some(i) => Self::new(i),
            },
            Self::Big(i) => Self::Big(i - rhs).shrink(),
        }
    }
}
impl Sub<&BigInt> for Integer {
    type Output = Integer;

    fn sub(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => {
                let i = BigInt::from(i);
                Self::Big(i - rhs).shrink()
            }
            Self::Big(i) => Self::Big(i - rhs).shrink(),
        }
    }
}
impl Sub<&Integer> for Integer {
    type Output = Integer;

    fn sub(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.sub(*rhs),
            Self::Big(rhs) => self.sub(rhs),
        }
    }
}
impl Sub<Integer> for Integer {
    type Output = Integer;

    fn sub(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.sub(rhs),
            Self::Big(rhs) => self.sub(&rhs),
        }
    }
}
impl Rem<i64> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn rem(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => match i.checked_rem(rhs) {
                None if rhs == 0 => Err(DivisionError),
                None => {
                    let i = BigInt::from(i);
                    Ok(Self::Big(i.rem(rhs)))
                }
                Some(i) => Ok(Self::new(i)),
            },
            Self::Big(_) if rhs == 0 => Err(DivisionError),
            Self::Big(i) => Ok(Self::Big(i.rem(rhs)).shrink()),
        }
    }
}
impl Rem<&BigInt> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn rem(self, rhs: &BigInt) -> Self::Output {
        if rhs.is_zero() {
            return Err(DivisionError);
        }
        match self {
            Self::Small(i) => {
                let i = BigInt::from(i);
                Ok(Self::Big(i.rem(rhs)).shrink())
            }
            Self::Big(i) => Ok(Self::Big(i.rem(rhs)).shrink()),
        }
    }
}
impl Rem<&Integer> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn rem(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.rem(*rhs),
            Self::Big(rhs) => self.rem(rhs),
        }
    }
}
impl Rem<Integer> for Integer {
    type Output = Result<Integer, DivisionError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.rem(rhs),
            Self::Big(rhs) => self.rem(&rhs),
        }
    }
}
impl BitAnd<i64> for Integer {
    type Output = Integer;

    fn bitand(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(i & rhs),
            Self::Big(i) => Self::Big(i & BigInt::from(rhs)),
        }
    }
}
impl BitAnd<&BigInt> for Integer {
    type Output = Integer;

    fn bitand(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => Self::Big(BigInt::from(i) & rhs).shrink(),
            Self::Big(i) => Self::Big(i & rhs).shrink(),
        }
    }
}
impl BitAnd<&Integer> for Integer {
    type Output = Integer;

    fn bitand(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitand(*rhs),
            Self::Big(rhs) => self.bitand(rhs),
        }
    }
}
impl BitAnd<Integer> for Integer {
    type Output = Integer;

    fn bitand(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitand(rhs),
            Self::Big(rhs) => self.bitand(&rhs),
        }
    }
}
impl BitOr<i64> for Integer {
    type Output = Integer;

    fn bitor(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(i | rhs),
            Self::Big(i) => Self::Big(i | BigInt::from(rhs)),
        }
    }
}
impl BitOr<&BigInt> for Integer {
    type Output = Integer;

    fn bitor(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => Self::Big(BigInt::from(i) | rhs),
            Self::Big(i) => Self::Big(i | rhs),
        }
    }
}
impl BitOr<&Integer> for Integer {
    type Output = Integer;

    fn bitor(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitor(*rhs),
            Self::Big(rhs) => self.bitor(rhs),
        }
    }
}
impl BitOr<Integer> for Integer {
    type Output = Integer;

    fn bitor(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitor(rhs),
            Self::Big(rhs) => self.bitor(&rhs),
        }
    }
}
impl BitXor<i64> for Integer {
    type Output = Integer;

    fn bitxor(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(i ^ rhs),
            Self::Big(i) => Self::Big(i ^ BigInt::from(rhs)),
        }
    }
}
impl BitXor<&BigInt> for Integer {
    type Output = Integer;

    fn bitxor(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => Self::Big(BigInt::from(i) ^ rhs),
            Self::Big(i) => Self::Big(i ^ rhs),
        }
    }
}
impl BitXor<&Integer> for Integer {
    type Output = Integer;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitxor(*rhs),
            Self::Big(rhs) => self.bitxor(rhs),
        }
    }
}
impl BitXor<Integer> for Integer {
    type Output = Integer;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitxor(rhs),
            Self::Big(rhs) => self.bitxor(&rhs),
        }
    }
}
impl Neg for Integer {
    type Output = Integer;

    fn neg(self) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(-i),
            Self::Big(i) => Self::Big(-i),
        }
    }
}
impl Not for Integer {
    type Output = Integer;

    fn not(self) -> Self::Output {
        match self {
            Self::Small(i) => Self::new(!i),
            Self::Big(i) => Self::Big(!i),
        }
    }
}
impl ToPrimitive for Integer {
    fn to_i64(&self) -> Option<i64> {
        match self {
            Self::Small(i) => Some(*i),
            Self::Big(i) => i.to_i64(),
        }
    }

    fn to_u64(&self) -> Option<u64> {
        match self {
            Self::Small(i) => i.to_u64(),
            Self::Big(i) => i.to_u64(),
        }
    }
}
impl FromPrimitive for Integer {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::new(n))
    }

    fn from_u64(n: u64) -> Option<Self> {
        if let Ok(int) = n.try_into() {
            Some(Self::new(int))
        } else {
            Some(Self::Big(n.into()))
        }
    }
}
impl From<u8> for Integer {
    fn from(i: u8) -> Self {
        Self::Small(i.into())
    }
}
impl From<u16> for Integer {
    fn from(i: u16) -> Self {
        Self::Small(i.into())
    }
}
impl From<u32> for Integer {
    fn from(i: u32) -> Self {
        Self::Small(i.into())
    }
}
impl From<i8> for Integer {
    fn from(i: i8) -> Self {
        Self::Small(i.into())
    }
}
impl From<i16> for Integer {
    fn from(i: i16) -> Self {
        Self::Small(i.into())
    }
}
impl From<i64> for Integer {
    fn from(i: i64) -> Self {
        Self::new(i)
    }
}
impl From<u64> for Integer {
    fn from(i: u64) -> Self {
        match i.try_into() {
            Ok(i) => Self::new(i),
            Err(_) => Self::Big(BigInt::from(i)),
        }
    }
}
impl From<i32> for Integer {
    fn from(i: i32) -> Self {
        Self::Small(i.into())
    }
}
impl From<usize> for Integer {
    fn from(i: usize) -> Self {
        match i.try_into() {
            Ok(i) => Self::new(i),
            Err(_) => Self::Big(BigInt::from(i)),
        }
    }
}
impl From<char> for Integer {
    fn from(i: char) -> Self {
        Self::Small(i as u32 as i64)
    }
}
impl From<BigInt> for Integer {
    fn from(i: BigInt) -> Self {
        Self::Big(i).shrink()
    }
}
impl TryInto<u8> for Integer {
    type Error = ();
    fn try_into(self) -> Result<u8, Self::Error> {
        match self {
            Self::Small(i) => i.try_into().map_err(|_| ()),
            Self::Big(_) => Err(()),
        }
    }
}
impl TryInto<i64> for Integer {
    type Error = ();
    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Self::Small(i) => Ok(i),
            Self::Big(i) => i.to_i64().ok_or(()),
        }
    }
}
impl TryInto<usize> for Integer {
    type Error = ();
    fn try_into(self) -> Result<usize, Self::Error> {
        match self {
            Self::Small(i) => i.try_into().map_err(|_| ()),
            Self::Big(i) => i.to_usize().ok_or(()),
        }
    }
}
