use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use core::str::FromStr;

pub use num_bigint::{ToBigInt, ToBigUint};
pub use num_integer::{ExtendedGcd, Integer};
pub use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

use num_bigint::{BigInt, BigUint, ParseBigIntError};

use crate::{DivisionError, Float, FloatError};

/// This struct unifies the fixed-width and aribtrary precision integral types in Firefly
#[derive(Debug, Clone, Hash)]
#[repr(u8)]
pub enum Int {
    Small(i64),
    Big(BigInt),
}
impl Int {
    // NOTE: See OpaqueTerm in liblumen_rt for the authoritative source of these constants
    const SIGN_BIT: u64 = 1 << 51;
    const NEG_INFINITY: u64 = unsafe { core::mem::transmute::<f64, u64>(f64::NEG_INFINITY) };
    const INTEGER_TAG: u64 = Self::NEG_INFINITY;
    const UNSIGNED_BITS: u64 = !(Self::INTEGER_TAG | Self::SIGN_BIT);
    pub const MAX_SMALL: i64 = (Self::UNSIGNED_BITS as i64);
    pub const MIN_SMALL: i64 = (!Self::UNSIGNED_BITS as i64);

    #[inline]
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
            Self::Small(num) => Self::new((*num).abs()),
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

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Small(int) => int.fmt(f),
            Self::Big(int) => int.fmt(f),
        }
    }
}

impl FromStr for Int {
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

impl Eq for Int {}
impl PartialEq for Int {
    fn eq(&self, rhs: &Int) -> bool {
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
impl PartialEq<f64> for Int {
    fn eq(&self, rhs: &f64) -> bool {
        match self {
            Self::Small(lhs) => (*lhs as f64).eq(rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).eq(rhs),
        }
    }
}
impl PartialEq<Float> for Int {
    fn eq(&self, rhs: &Float) -> bool {
        match self {
            Self::Small(lhs) => (*lhs as f64) == rhs.inner(),
            Self::Big(lhs) => crate::bigint_to_double(lhs) == rhs.inner(),
        }
    }
}
impl PartialEq<Int> for f64 {
    fn eq(&self, rhs: &Int) -> bool {
        rhs.eq(self)
    }
}
impl PartialEq<char> for Int {
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
impl PartialEq<Int> for char {
    fn eq(&self, rhs: &Int) -> bool {
        rhs.eq(self)
    }
}
impl PartialEq<i64> for Int {
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
impl PartialEq<Int> for i64 {
    fn eq(&self, rhs: &Int) -> bool {
        rhs.eq(self)
    }
}

impl Ord for Int {
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
impl PartialOrd for Int {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}
impl PartialOrd<f64> for Int {
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        match self {
            Self::Small(lhs) => (*lhs as f64).partial_cmp(rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).partial_cmp(rhs),
        }
    }
}
impl PartialOrd<Float> for Int {
    fn partial_cmp(&self, rhs: &Float) -> Option<Ordering> {
        let rhs = rhs.inner();
        match self {
            Self::Small(lhs) => (*lhs as f64).partial_cmp(&rhs),
            Self::Big(lhs) => crate::bigint_to_double(lhs).partial_cmp(&rhs),
        }
    }
}
impl PartialOrd<Int> for f64 {
    fn partial_cmp(&self, rhs: &Int) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl PartialOrd<char> for Int {
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
impl PartialOrd<Int> for char {
    fn partial_cmp(&self, rhs: &Int) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}
impl PartialOrd<i64> for Int {
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
impl PartialOrd<Int> for i64 {
    fn partial_cmp(&self, rhs: &Int) -> Option<Ordering> {
        rhs.partial_cmp(self).map(|v| v.reverse())
    }
}

impl Add for Int {
    type Output = Int;

    fn add(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.add(rhs),
            Self::Big(rhs) => self.add(&rhs),
        }
    }
}
impl Add<&Int> for Int {
    type Output = Int;

    fn add(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.add(*rhs),
            Self::Big(rhs) => self.add(rhs),
        }
    }
}
impl Add<i64> for Int {
    type Output = Int;

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
impl Add<&BigInt> for Int {
    type Output = Int;

    fn add(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(i) => (rhs + i).into(),
            Self::Big(i) => (i + rhs).into(),
        }
    }
}
impl Add<Float> for Int {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: Float) -> Self::Output {
        self.to_efloat()? + rhs
    }
}
impl Add<Float> for &Int {
    type Output = Result<Float, FloatError>;

    fn add(self, rhs: Float) -> Self::Output {
        self.to_efloat()? + rhs
    }
}

impl Sub for Int {
    type Output = Int;

    fn sub(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.sub(rhs),
            Self::Big(rhs) => self.sub(&rhs),
        }
    }
}
impl Sub<&Int> for Int {
    type Output = Int;

    fn sub(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.sub(*rhs),
            Self::Big(rhs) => self.sub(rhs),
        }
    }
}
impl Sub<i64> for Int {
    type Output = Int;

    fn sub(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => match lhs.checked_sub(rhs) {
                None => {
                    let lhs = BigInt::from(lhs);
                    (lhs - rhs).into()
                }
                Some(result) => result.into(),
            },
            Self::Big(lhs) => (lhs - rhs).into(),
        }
    }
}
impl Sub<&BigInt> for Int {
    type Output = Int;

    fn sub(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                (lhs - rhs).into()
            }
            Self::Big(lhs) => (lhs - rhs).into(),
        }
    }
}
impl Sub<Float> for Int {
    type Output = Result<Float, FloatError>;
    fn sub(self, rhs: Float) -> Self::Output {
        self.to_efloat()? - rhs
    }
}
impl Sub<Float> for &Int {
    type Output = Result<Float, FloatError>;
    fn sub(self, rhs: Float) -> Self::Output {
        self.to_efloat()? - rhs
    }
}

impl Mul for Int {
    type Output = Int;

    fn mul(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.mul(rhs),
            Self::Big(rhs) => self.mul(&rhs),
        }
    }
}
impl Mul<&Int> for Int {
    type Output = Int;
    fn mul(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.mul(*rhs),
            Self::Big(rhs) => self.mul(rhs),
        }
    }
}
impl Mul<usize> for &Int {
    type Output = Int;
    fn mul(self, rhs: usize) -> Self::Output {
        let rhs64: Result<i64, _> = rhs.try_into();
        match rhs64 {
            Ok(rhs) => match self {
                Int::Small(lhs) => match lhs.checked_mul(rhs) {
                    None => {
                        let lhs = BigInt::from(*lhs);
                        (lhs * rhs).into()
                    }
                    Some(result) => result.into(),
                },
                Int::Big(lhs) => (lhs * rhs).into(),
            },
            Err(_) => match self {
                Int::Small(lhs) => {
                    let lhs = BigInt::from(*lhs);
                    (lhs * rhs).into()
                }
                Int::Big(lhs) => (lhs * rhs).into(),
            },
        }
    }
}
impl Mul<i64> for Int {
    type Output = Int;
    fn mul(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => match lhs.checked_mul(rhs) {
                None => {
                    let lhs = BigInt::from(lhs);
                    (lhs * rhs).into()
                }
                Some(result) => result.into(),
            },
            Self::Big(lhs) => (lhs * rhs).into(),
        }
    }
}
impl Mul<&BigInt> for Int {
    type Output = Int;
    fn mul(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                (lhs * rhs).into()
            }
            Self::Big(lhs) => (lhs * rhs).into(),
        }
    }
}
impl Mul<Float> for Int {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: Float) -> Self::Output {
        self.to_efloat()? * rhs
    }
}
impl Mul<Float> for &Int {
    type Output = Result<Float, FloatError>;
    fn mul(self, rhs: Float) -> Self::Output {
        self.to_efloat()? * rhs
    }
}

impl Div for Int {
    type Output = Result<Int, DivisionError>;

    fn div(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.div(rhs),
            Self::Big(rhs) => self.div(&rhs),
        }
    }
}
impl Div<&Int> for Int {
    type Output = Result<Int, DivisionError>;

    fn div(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.div(*rhs),
            Self::Big(rhs) => self.div(rhs),
        }
    }
}
impl Div<i64> for Int {
    type Output = Result<Int, DivisionError>;

    fn div(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => match lhs.checked_div(rhs) {
                None if rhs == 0 => Err(DivisionError),
                None => {
                    let lhs = BigInt::from(lhs);
                    Ok((lhs / rhs).into())
                }
                Some(result) => Ok(result.into()),
            },
            Self::Big(_) if rhs == 0 => Err(DivisionError),
            Self::Big(lhs) => Ok((lhs / rhs).into()),
        }
    }
}
impl Div<&BigInt> for Int {
    type Output = Result<Int, DivisionError>;

    fn div(self, rhs: &BigInt) -> Self::Output {
        if rhs.is_zero() {
            return Err(DivisionError);
        }

        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                Ok((lhs / rhs).into())
            }
            Self::Big(lhs) => Ok((lhs / rhs).into()),
        }
    }
}
impl Div<Float> for Int {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: Float) -> Self::Output {
        self.to_efloat().map_err(|_| DivisionError)? / rhs
    }
}
impl Div<Float> for &Int {
    type Output = Result<Float, DivisionError>;

    fn div(self, rhs: Float) -> Self::Output {
        self.to_efloat().map_err(|_| DivisionError)? / rhs
    }
}

impl Rem for Int {
    type Output = Result<Int, DivisionError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.rem(rhs),
            Self::Big(rhs) => self.rem(&rhs),
        }
    }
}
impl Rem<&Int> for Int {
    type Output = Result<Int, DivisionError>;

    fn rem(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.rem(*rhs),
            Self::Big(rhs) => self.rem(rhs),
        }
    }
}
impl Rem<i64> for Int {
    type Output = Result<Int, DivisionError>;

    fn rem(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => match lhs.checked_rem(rhs) {
                None if rhs == 0 => Err(DivisionError),
                None => {
                    let lhs = BigInt::from(lhs);
                    Ok((lhs % rhs).into())
                }
                Some(result) => Ok(result.into()),
            },
            Self::Big(_) if rhs == 0 => Err(DivisionError),
            Self::Big(lhs) => Ok((lhs % rhs).into()),
        }
    }
}
impl Rem<&BigInt> for Int {
    type Output = Result<Int, DivisionError>;

    fn rem(self, rhs: &BigInt) -> Self::Output {
        if rhs.is_zero() {
            return Err(DivisionError);
        }

        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                Ok((lhs % rhs).into())
            }
            Self::Big(lhs) => Ok((lhs % rhs).into()),
        }
    }
}
impl Rem<Float> for Int {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: Float) -> Self::Output {
        self.to_efloat().map_err(|_| DivisionError)? % rhs
    }
}
impl Rem<Float> for &Int {
    type Output = Result<Float, DivisionError>;

    fn rem(self, rhs: Float) -> Self::Output {
        self.to_efloat().map_err(|_| DivisionError)? % rhs
    }
}

impl Shl<u32> for Int {
    type Output = Int;
    fn shl(self, y: u32) -> Self::Output {
        match self {
            Self::Small(x) => {
                let x = BigInt::from(x);
                (x << y).into()
            }
            Self::Big(x) => Self::Big(x << y),
        }
    }
}
impl Shl<Int> for Int {
    type Output = Int;

    fn shl(self, num: Int) -> Self::Output {
        match (self, num) {
            (Self::Small(x), Self::Small(y)) => {
                let y32: Result<u32, _> = y.try_into();
                match y32 {
                    Ok(y) => match x.checked_shl(y) {
                        None => {
                            let x = BigInt::from(x);
                            (x << y).into()
                        }
                        Some(result) => result.into(),
                    },
                    Err(_) => {
                        let x = BigInt::from(x);
                        (x << y).into()
                    }
                }
            }
            (Self::Small(x), Self::Big(y)) => {
                let x = BigInt::from(x);
                match y.bits() {
                    n if n <= 64 => {
                        let y: u64 = y.to_u64().unwrap();
                        (x << y).into()
                    }
                    n if n <= 128 => {
                        let y: u128 = y.to_u128().unwrap();
                        (x << y).into()
                    }
                    n => panic!("invalid shift, value is too large: {}", n),
                }
            }
            (Self::Big(x), Self::Small(y)) => (x << y).into(),
            (Self::Big(x), Self::Big(y)) => match y.bits() {
                n if n <= 64 => {
                    let y: u64 = y.to_u64().unwrap();
                    (x << y).into()
                }
                n if n <= 128 => {
                    let y: u128 = y.to_u128().unwrap();
                    (x << y).into()
                }
                n => panic!("invalid shift, value is too large: {}", n),
            },
        }
    }
}

impl Shr<u32> for Int {
    type Output = Int;
    fn shr(self, y: u32) -> Self::Output {
        match self {
            Self::Small(x) => (x >> y).into(),
            Self::Big(x) => (x >> y).into(),
        }
    }
}
impl Shr<Int> for Int {
    type Output = Int;
    fn shr(self, num: Int) -> Self::Output {
        match (self, num) {
            (Self::Small(x), Self::Small(y)) => {
                let y32: Result<u32, _> = y.try_into();
                match y32 {
                    Ok(y) => (x >> y).into(),
                    Err(_) => {
                        let x = BigInt::from(x);
                        (x >> y).into()
                    }
                }
            }
            (Self::Small(x), Self::Big(y)) => {
                let x = BigInt::from(x);
                match y.bits() {
                    n if n <= 64 => {
                        let y: u64 = y.to_u64().unwrap();
                        (x >> y).into()
                    }
                    _ => Int::Small(0),
                }
            }
            (Self::Big(x), Self::Small(y)) => (x >> y).into(),
            (Self::Big(x), Self::Big(y)) => match y.bits() {
                n if n <= 64 => {
                    let y: u64 = y.to_u64().unwrap();
                    (x >> y).into()
                }
                _ => Int::Small(0),
            },
        }
    }
}

impl BitAnd for Int {
    type Output = Int;

    fn bitand(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitand(rhs),
            Self::Big(rhs) => self.bitand(&rhs),
        }
    }
}
impl BitAnd<&Int> for Int {
    type Output = Int;

    fn bitand(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitand(*rhs),
            Self::Big(rhs) => self.bitand(rhs),
        }
    }
}
impl BitAnd<i64> for Int {
    type Output = Int;

    fn bitand(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => (lhs & rhs).into(),
            Self::Big(lhs) => {
                let rhs = BigInt::from(rhs);
                (lhs & rhs).into()
            }
        }
    }
}
impl BitAnd<&BigInt> for Int {
    type Output = Int;

    fn bitand(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                (lhs & rhs).into()
            }
            Self::Big(lhs) => (lhs & rhs).into(),
        }
    }
}

impl BitOr for Int {
    type Output = Int;

    fn bitor(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitor(rhs),
            Self::Big(rhs) => self.bitor(&rhs),
        }
    }
}
impl BitOr<&Int> for Int {
    type Output = Int;

    fn bitor(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitor(*rhs),
            Self::Big(rhs) => self.bitor(rhs),
        }
    }
}
impl BitOr<i64> for Int {
    type Output = Int;

    fn bitor(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => (lhs | rhs).into(),
            Self::Big(lhs) => {
                let rhs = BigInt::from(rhs);
                (lhs | rhs).into()
            }
        }
    }
}
impl BitOr<&BigInt> for Int {
    type Output = Int;

    fn bitor(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                (lhs | rhs).into()
            }
            Self::Big(lhs) => (lhs | rhs).into(),
        }
    }
}

impl BitXor for Int {
    type Output = Int;

    fn bitxor(self, rhs: Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitxor(rhs),
            Self::Big(rhs) => self.bitxor(&rhs),
        }
    }
}
impl BitXor<&Int> for Int {
    type Output = Int;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        match rhs {
            Self::Small(rhs) => self.bitxor(*rhs),
            Self::Big(rhs) => self.bitxor(rhs),
        }
    }
}
impl BitXor<i64> for Int {
    type Output = Int;

    fn bitxor(self, rhs: i64) -> Self::Output {
        match self {
            Self::Small(lhs) => (lhs ^ rhs).into(),
            Self::Big(lhs) => {
                let rhs = BigInt::from(rhs);
                (lhs ^ rhs).into()
            }
        }
    }
}
impl BitXor<&BigInt> for Int {
    type Output = Int;

    fn bitxor(self, rhs: &BigInt) -> Self::Output {
        match self {
            Self::Small(lhs) => {
                let lhs = BigInt::from(lhs);
                (lhs ^ rhs).into()
            }
            Self::Big(lhs) => (lhs ^ rhs).into(),
        }
    }
}

impl Neg for Int {
    type Output = Int;

    fn neg(self) -> Self::Output {
        match self {
            Self::Small(i) => (-i).into(),
            Self::Big(i) => (-i).into(),
        }
    }
}

impl Not for Int {
    type Output = Int;

    fn not(self) -> Self::Output {
        match self {
            Self::Small(i) => (!i).into(),
            Self::Big(i) => (!i).into(),
        }
    }
}

impl ToBigInt for Int {
    fn to_bigint(&self) -> Option<BigInt> {
        match self {
            Self::Small(int) => Some(BigInt::from(*int)),
            Self::Big(num) => Some(num.clone()),
        }
    }
}
impl ToBigUint for Int {
    fn to_biguint(&self) -> Option<BigUint> {
        match self {
            Self::Small(i) if *i >= 0 => Some(BigUint::from(*i as u64)),
            Self::Small(_) => None,
            Self::Big(i) => i.to_biguint(),
        }
    }
}

impl ToPrimitive for Int {
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

impl FromPrimitive for Int {
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

impl From<u8> for Int {
    #[inline(always)]
    fn from(i: u8) -> Self {
        Self::Small(i.into())
    }
}
impl From<u16> for Int {
    #[inline(always)]
    fn from(i: u16) -> Self {
        Self::Small(i.into())
    }
}
impl From<u32> for Int {
    #[inline(always)]
    fn from(i: u32) -> Self {
        Self::Small(i.into())
    }
}
impl From<i8> for Int {
    #[inline(always)]
    fn from(i: i8) -> Self {
        Self::Small(i.into())
    }
}
impl From<i16> for Int {
    #[inline(always)]
    fn from(i: i16) -> Self {
        Self::Small(i.into())
    }
}
impl From<i64> for Int {
    #[inline(always)]
    fn from(i: i64) -> Self {
        Self::new(i)
    }
}
impl From<u64> for Int {
    #[inline(always)]
    fn from(i: u64) -> Self {
        unsafe { Int::from_u64(i).unwrap_unchecked() }
    }
}
impl From<i32> for Int {
    #[inline(always)]
    fn from(i: i32) -> Self {
        Self::Small(i.into())
    }
}
impl From<usize> for Int {
    #[inline(always)]
    fn from(i: usize) -> Self {
        unsafe { Int::from_usize(i).unwrap_unchecked() }
    }
}
impl From<i128> for Int {
    fn from(i: i128) -> Self {
        if i > Self::MAX_SMALL as i128 || i < Self::MIN_SMALL as i128 {
            Self::Big(i.into())
        } else {
            Self::Small(i as i64)
        }
    }
}
impl From<char> for Int {
    #[inline]
    fn from(i: char) -> Self {
        Self::Small(i as i64)
    }
}
impl From<BigInt> for Int {
    #[inline]
    fn from(i: BigInt) -> Self {
        match i.to_i64() {
            Some(n) if n <= Self::MAX_SMALL && n >= Self::MIN_SMALL => Self::Small(n),
            Some(_) | None => Self::Big(i),
        }
    }
}
impl TryInto<u8> for Int {
    type Error = ();
    fn try_into(self) -> Result<u8, Self::Error> {
        match self {
            Self::Small(i) => i.try_into().map_err(|_| ()),
            Self::Big(i) => i.to_u8().ok_or(()),
        }
    }
}
impl TryInto<i64> for Int {
    type Error = ();
    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Self::Small(i) => Ok(i),
            Self::Big(i) => i.to_i64().ok_or(()),
        }
    }
}
impl TryInto<usize> for Int {
    type Error = ();
    fn try_into(self) -> Result<usize, Self::Error> {
        match self {
            Self::Small(i) => i.try_into().map_err(|_| ()),
            Self::Big(i) => i.to_usize().ok_or(()),
        }
    }
}
