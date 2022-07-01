use crate::{DivisionError, Float, FloatError};

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use core::str::FromStr;

pub use num_bigint::ToBigInt;
use num_bigint::{BigInt, ParseBigIntError, Sign};
pub use num_traits::{FromPrimitive, ToPrimitive, Zero};

#[derive(Debug, Clone, Hash)]
pub enum Integer {
    Small(i64),
    Big(BigInt),
}

impl ToBigInt for Integer {
    fn to_bigint(&self) -> Option<BigInt> {
        match self {
            Integer::Small(int) => Some(BigInt::from(*int)),
            Integer::Big(num) => Some(num.clone()),
        }
    }
}

impl Integer {
    pub fn is_zero(&self) -> bool {
        match self {
            Integer::Small(num) => *num == 0,
            Integer::Big(num) => num.is_zero(),
        }
    }

    pub fn plus(&self) -> Integer {
        match self {
            Integer::Small(num) if num < &0 => Integer::Small(-num),
            Integer::Big(num) if num.sign() == Sign::Minus => Integer::Big(-num.clone()),
            _ => self.clone(),
        }
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Integer::Small(int) => *int as f64,
            Integer::Big(int) => crate::bigint_to_double(int),
        }
    }

    pub fn to_efloat(&self) -> Result<Float, FloatError> {
        Float::new(self.to_float())
    }

    pub fn shrink(self) -> Self {
        match self {
            Integer::Small(int) => Integer::Small(int),
            Integer::Big(int) => {
                if let Some(small) = int.to_i64() {
                    Integer::Small(small)
                } else {
                    Integer::Big(int)
                }
            }
        }
    }

    pub fn from_string_radix(string: &str, radix: u32) -> Option<Integer> {
        if let Ok(i) = i64::from_str_radix(string, radix) {
            return Some(Integer::Small(i));
        }
        let bi = BigInt::parse_bytes(string.as_bytes(), radix)?;
        Some(Integer::Big(bi))
    }

    pub fn to_u64(&self) -> Option<u64> {
        ToPrimitive::to_u64(self)
    }

    pub fn to_usize(&self) -> Option<usize> {
        ToPrimitive::to_usize(self)
    }

    pub fn to_arity(&self) -> u8 {
        match self {
            Integer::Small(i) => (*i).try_into().unwrap(),
            Integer::Big(_) => {
                panic!("invalid arity, expected value within u8 range, but got big integer")
            }
        }
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Integer::Small(int) => int.fmt(f),
            Integer::Big(int) => int.fmt(f),
        }
    }
}

impl FromStr for Integer {
    type Err = ParseBigIntError;
    fn from_str(s: &str) -> Result<Self, ParseBigIntError> {
        match s.parse::<i64>() {
            Ok(int) => Ok(Integer::Small(int)),
            Err(_) => match s.parse::<BigInt>() {
                Ok(int) => Ok(Integer::Big(int)),
                Err(err) => Err(err),
            },
        }
    }
}

impl PartialEq for Integer {
    fn eq(&self, rhs: &Integer) -> bool {
        match (self, rhs) {
            (Integer::Small(lhs), Integer::Small(rhs)) => lhs.eq(rhs),
            (Integer::Small(lhs), Integer::Big(rhs)) => {
                if let Some(ref i) = rhs.to_i64() {
                    return lhs.eq(i);
                }
                false
            }
            (Integer::Big(lhs), Integer::Small(rhs)) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.eq(rhs);
                }
                false
            }
            (Integer::Big(lhs), Integer::Big(rhs)) => lhs.eq(rhs),
        }
    }
}
impl Eq for Integer {}

impl PartialOrd for Integer {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        match (self, rhs) {
            (Integer::Small(lhs), Integer::Small(rhs)) => lhs.partial_cmp(rhs),
            (Integer::Small(lhs), Integer::Big(rhs)) => {
                if let Some(ref i) = rhs.to_i64() {
                    return lhs.partial_cmp(i);
                }
                Some(if rhs.sign() == Sign::Minus {
                    Ordering::Greater
                } else {
                    Ordering::Less
                })
            }
            (Integer::Big(lhs), Integer::Small(rhs)) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.partial_cmp(rhs);
                }
                Some(if lhs.sign() == Sign::Minus {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
            }
            (Integer::Big(lhs), Integer::Big(rhs)) => lhs.partial_cmp(rhs),
        }
    }
}
impl Ord for Integer {
    fn cmp(&self, rhs: &Integer) -> Ordering {
        self.partial_cmp(rhs).unwrap()
    }
}

impl PartialEq<f64> for Integer {
    fn eq(&self, rhs: &f64) -> bool {
        match self {
            Integer::Small(lhs) => (*lhs as f64).eq(rhs),
            Integer::Big(lhs) => crate::bigint_to_double(lhs).eq(rhs),
        }
    }
}
impl PartialEq<Integer> for f64 {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}
impl PartialOrd<f64> for Integer {
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        match self {
            Integer::Small(lhs) => (*lhs as f64).partial_cmp(rhs),
            Integer::Big(lhs) => crate::bigint_to_double(lhs).partial_cmp(rhs),
        }
    }
}
impl PartialOrd<Integer> for f64 {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl PartialEq<char> for Integer {
    fn eq(&self, rhs: &char) -> bool {
        match self {
            Integer::Small(lhs) => lhs.eq(&(*rhs as i64)),
            Integer::Big(lhs) => {
                let rhs = *rhs as i64;
                if let Some(ref i) = lhs.to_i64() {
                    return i.eq(&rhs);
                }
                false
            }
        }
    }
}
impl PartialEq<Integer> for char {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}
impl PartialOrd<char> for Integer {
    fn partial_cmp(&self, rhs: &char) -> Option<Ordering> {
        match self {
            Integer::Small(lhs) => lhs.partial_cmp(&(*rhs as i64)),
            Integer::Big(lhs) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.partial_cmp(&(*rhs as i64));
                }
                Some(if lhs.sign() == Sign::Minus {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
            }
        }
    }
}
impl PartialOrd<Integer> for char {
    fn partial_cmp(&self, rhs: &Integer) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl PartialEq<i64> for Integer {
    fn eq(&self, rhs: &i64) -> bool {
        match self {
            Integer::Small(lhs) => lhs.eq(rhs),
            Integer::Big(lhs) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.eq(rhs);
                }
                false
            }
        }
    }
}
impl PartialEq<Integer> for i64 {
    fn eq(&self, rhs: &Integer) -> bool {
        rhs.eq(self)
    }
}
impl PartialOrd<i64> for Integer {
    fn partial_cmp(&self, rhs: &i64) -> Option<Ordering> {
        match self {
            Integer::Small(lhs) => lhs.partial_cmp(rhs),
            Integer::Big(lhs) => {
                if let Some(ref i) = lhs.to_i64() {
                    return i.partial_cmp(rhs);
                }
                Some(if lhs.sign() == Sign::Minus {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
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
    fn shr(self, num: u32) -> Integer {
        let big = self.to_bigint().unwrap() >> (num as usize);
        Integer::Big(big).shrink()
    }
}
impl Shl<u32> for Integer {
    type Output = Integer;
    fn shl(self, num: u32) -> Integer {
        let big = self.to_bigint().unwrap() << (num as usize);
        Integer::Big(big).shrink()
    }
}

impl Mul<i64> for Integer {
    type Output = Integer;
    fn mul(self, rhs: i64) -> Integer {
        match self {
            Integer::Small(lhs) => {
                let mut int: BigInt = lhs.into();
                int = int * rhs;
                Integer::Big(int).shrink()
            }
            Integer::Big(lhs) => Integer::Big(lhs * rhs),
        }
    }
}
impl Mul<&Integer> for Integer {
    type Output = Integer;
    fn mul(self, rhs: &Integer) -> Integer {
        let mut lhs = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(rhs) => lhs = lhs * rhs,
            Integer::Big(rhs) => lhs *= rhs,
        }
        Integer::Big(lhs).shrink()
    }
}
impl Div<&Integer> for Integer {
    type Output = Result<Integer, DivisionError>;
    fn div(self, rhs: &Integer) -> Self::Output {
        if rhs.is_zero() {
            return Err(DivisionError);
        }

        let mut lhs = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(rhs) => lhs = lhs / rhs,
            Integer::Big(rhs) => lhs /= rhs,
        }
        Ok(Integer::Big(lhs).shrink())
    }
}
impl Add<&Integer> for Integer {
    type Output = Integer;
    fn add(self, rhs: &Integer) -> Integer {
        let mut lhs = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(rhs) => lhs = lhs + rhs,
            Integer::Big(rhs) => lhs += rhs,
        }
        Integer::Big(lhs).shrink()
    }
}
impl Sub<&Integer> for Integer {
    type Output = Integer;
    fn sub(self, rhs: &Integer) -> Integer {
        let mut lhs = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(rhs) => lhs = lhs - rhs,
            Integer::Big(rhs) => lhs -= rhs,
        }
        Integer::Big(lhs).shrink()
    }
}
impl Rem<&Integer> for Integer {
    type Output = Integer;
    fn rem(self, rhs: &Integer) -> Integer {
        let mut lhs = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(rhs) => lhs = lhs % rhs,
            Integer::Big(rhs) => lhs %= rhs,
        }
        Integer::Big(lhs).shrink()
    }
}

impl BitAnd<&Integer> for Integer {
    type Output = Integer;
    fn bitand(self, rhs: &Integer) -> Integer {
        let l = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(r) => l & BigInt::from(*r),
            Integer::Big(r) => l & r,
        }
        .into()
    }
}
impl BitOr<&Integer> for Integer {
    type Output = Integer;
    fn bitor(self, rhs: &Integer) -> Integer {
        let l = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(r) => l | BigInt::from(*r),
            Integer::Big(r) => l | r,
        }
        .into()
    }
}
impl BitXor<&Integer> for Integer {
    type Output = Integer;
    fn bitxor(self, rhs: &Integer) -> Integer {
        let l = self.to_bigint().unwrap();
        match rhs {
            Integer::Small(r) => l ^ BigInt::from(*r),
            Integer::Big(r) => l ^ r,
        }
        .into()
    }
}

impl Neg for Integer {
    type Output = Integer;
    fn neg(self) -> Integer {
        match self {
            Integer::Small(int) => Integer::Small(-int),
            Integer::Big(int) => Integer::Big(-int),
        }
    }
}
impl Not for &Integer {
    type Output = Integer;
    fn not(self) -> Integer {
        match self {
            Integer::Small(int) => Integer::Small(!int),
            Integer::Big(int) => Integer::Big(!int),
        }
    }
}

impl ToPrimitive for Integer {
    fn to_i64(&self) -> Option<i64> {
        match self {
            Integer::Small(int) => int.to_i64(),
            Integer::Big(int) => int.to_i64(),
        }
    }
    fn to_u64(&self) -> Option<u64> {
        match self {
            Integer::Small(int) => int.to_u64(),
            Integer::Big(int) => int.to_u64(),
        }
    }
}

impl FromPrimitive for Integer {
    fn from_i64(n: i64) -> Option<Integer> {
        Some(Integer::Small(n))
    }
    fn from_u64(n: u64) -> Option<Integer> {
        if let Ok(int) = n.try_into() {
            Some(Integer::Small(int))
        } else {
            Some(Integer::Big(n.into()))
        }
    }
}
impl From<u8> for Integer {
    fn from(i: u8) -> Integer {
        Integer::Small(i.into())
    }
}
impl From<u16> for Integer {
    fn from(i: u16) -> Integer {
        Integer::Small(i.into())
    }
}
impl From<u32> for Integer {
    fn from(i: u32) -> Integer {
        Integer::from_u32(i).unwrap()
    }
}
impl From<i8> for Integer {
    fn from(i: i8) -> Integer {
        Integer::Small(i.into())
    }
}
impl From<i16> for Integer {
    fn from(i: i16) -> Integer {
        Integer::Small(i.into())
    }
}
impl From<i64> for Integer {
    fn from(i: i64) -> Integer {
        Integer::from_i64(i).unwrap()
    }
}
impl From<u64> for Integer {
    fn from(i: u64) -> Integer {
        Integer::from_u64(i).unwrap()
    }
}
impl From<i32> for Integer {
    fn from(i: i32) -> Integer {
        Integer::from_i32(i).unwrap()
    }
}
impl From<usize> for Integer {
    fn from(i: usize) -> Integer {
        Integer::from_usize(i).unwrap()
    }
}
impl From<char> for Integer {
    fn from(i: char) -> Integer {
        Integer::from_u64(i as u64).unwrap()
    }
}
impl From<BigInt> for Integer {
    fn from(i: BigInt) -> Integer {
        Integer::Big(i)
    }
}
impl TryInto<u8> for Integer {
    type Error = ();
    fn try_into(self) -> Result<u8, Self::Error> {
        match self {
            Integer::Small(i) => i.try_into().map_err(|_| ()),
            Integer::Big(_) => Err(()),
        }
    }
}
