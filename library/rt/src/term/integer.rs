use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use core::ops::{Deref, DerefMut};

use firefly_alloc::clone::WriteCloneIntoRaw;
use firefly_alloc::heap::Heap;
use firefly_number::{ExtendedGcd, Float, Integer, Num, One, Pow, Signed, ToPrimitive, Zero};

use crate::gc::Gc;

use super::{Boxable, Header, Tag};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BigInt {
    header: Header,
    value: firefly_number::BigInt,
}
impl Default for BigInt {
    fn default() -> Self {
        Self {
            header: Header::new(Tag::BigInt, 0),
            value: Zero::zero(),
        }
    }
}
impl BigInt {
    pub fn new<I: Into<firefly_number::BigInt>>(value: I) -> Self {
        Self {
            header: Header::new(Tag::BigInt, 0),
            value: value.into(),
        }
    }

    #[inline]
    pub const fn inner(&self) -> &firefly_number::BigInt {
        &self.value
    }

    pub fn abs(&self) -> Self {
        let value = self.value.abs();
        Self {
            header: self.header,
            value,
        }
    }
}
impl Boxable for BigInt {
    type Metadata = ();

    const TAG: Tag = Tag::BigInt;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Gc::new_uninit_in(heap).unwrap();
            unsafe {
                self.write_clone_into_raw(cloned.as_mut_ptr());
                cloned.assume_init()
            }
        }
    }
}
impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}
impl Deref for BigInt {
    type Target = firefly_number::BigInt;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl DerefMut for BigInt {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}
impl Eq for BigInt {}
impl PartialEq for BigInt {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}
impl PartialEq<Gc<BigInt>> for BigInt {
    fn eq(&self, other: &Gc<BigInt>) -> bool {
        self.value.eq(&other.value)
    }
}
impl PartialEq<i64> for BigInt {
    #[inline]
    fn eq(&self, other: &i64) -> bool {
        let rhs = firefly_number::BigInt::from(*other);
        self.value.eq(&rhs)
    }
}
impl PartialEq<Float> for BigInt {
    #[inline]
    fn eq(&self, other: &Float) -> bool {
        firefly_number::bigint_to_double(&self.value) == other.inner()
    }
}
impl PartialEq<BigInt> for Float {
    #[inline]
    fn eq(&self, other: &BigInt) -> bool {
        self.eq(&other.value)
    }
}
impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd<i64> for BigInt {
    fn partial_cmp(&self, other: &i64) -> Option<core::cmp::Ordering> {
        let rhs = firefly_number::BigInt::from(*other);
        Some(self.value.cmp(&rhs))
    }
}
impl PartialOrd<Float> for BigInt {
    fn partial_cmp(&self, other: &Float) -> Option<core::cmp::Ordering> {
        firefly_number::bigint_to_double(&self.value).partial_cmp(&other.inner())
    }
}
impl PartialOrd<BigInt> for Float {
    fn partial_cmp(&self, other: &BigInt) -> Option<core::cmp::Ordering> {
        self.partial_cmp(&other.value)
    }
}
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}
impl Hash for BigInt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}
impl<T: firefly_number::ToBigInt> From<T> for BigInt {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value.to_bigint().unwrap())
    }
}
impl Into<firefly_number::BigInt> for BigInt {
    fn into(self) -> firefly_number::BigInt {
        self.value
    }
}
impl TryInto<usize> for BigInt {
    type Error = ();
    fn try_into(self) -> Result<usize, Self::Error> {
        self.value.to_usize().ok_or(())
    }
}
impl<'a> TryInto<usize> for &'a BigInt {
    type Error = ();
    fn try_into(self) -> Result<usize, Self::Error> {
        self.value.to_usize().ok_or(())
    }
}
impl Num for BigInt {
    type FromStrRadixErr = <firefly_number::BigInt as Num>::FromStrRadixErr;

    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let value = firefly_number::BigInt::from_str_radix(s, radix)?;
        Ok(Self {
            header: Header::new(Tag::BigInt, 0),
            value,
        })
    }
}
impl Integer for BigInt {
    fn gcd(&self, other: &Self) -> Self {
        let value = self.value.gcd(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn lcm(&self, other: &Self) -> Self {
        let value = self.value.lcm(&other.value);
        Self {
            header: self.header,
            value,
        }
    }
    fn gcd_lcm(&self, other: &Self) -> (Self, Self) {
        let (l, r) = self.value.gcd_lcm(&other.value);
        (
            Self {
                header: self.header,
                value: l,
            },
            Self {
                header: self.header,
                value: r,
            },
        )
    }

    fn extended_gcd_lcm(&self, other: &Self) -> (ExtendedGcd<Self>, Self) {
        let (l, r) = self.value.extended_gcd_lcm(&other.value);
        let extended = ExtendedGcd {
            gcd: Self {
                header: self.header,
                value: l.gcd,
            },
            x: Self {
                header: self.header,
                value: l.x,
            },
            y: Self {
                header: self.header,
                value: l.y,
            },
        };
        (
            extended,
            Self {
                header: self.header,
                value: r,
            },
        )
    }

    fn divides(&self, other: &Self) -> bool {
        self.value.divides(&other.value)
    }

    fn is_multiple_of(&self, other: &Self) -> bool {
        self.value.is_multiple_of(&other.value)
    }

    fn is_even(&self) -> bool {
        self.value.is_even()
    }

    fn is_odd(&self) -> bool {
        self.value.is_odd()
    }

    fn next_multiple_of(&self, other: &Self) -> Self {
        let value = self.value.next_multiple_of(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn prev_multiple_of(&self, other: &Self) -> Self {
        let value = self.value.prev_multiple_of(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.value.div_rem(&other.value);
        (
            Self {
                header: self.header,
                value: q,
            },
            Self {
                header: self.header,
                value: r,
            },
        )
    }

    fn div_floor(&self, other: &Self) -> Self {
        let value = self.value.div_floor(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn mod_floor(&self, other: &Self) -> Self {
        let value = self.value.mod_floor(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn div_mod_floor(&self, other: &Self) -> (Self, Self) {
        let (q, r) = self.value.div_mod_floor(&other.value);
        (
            Self {
                header: self.header,
                value: q,
            },
            Self {
                header: self.header,
                value: r,
            },
        )
    }

    fn div_ceil(&self, other: &Self) -> Self {
        let value = self.value.div_ceil(&other.value);
        Self {
            header: self.header,
            value,
        }
    }

    fn extended_gcd(&self, other: &Self) -> ExtendedGcd<Self>
    where
        Self: Clone,
    {
        let extended = self.value.extended_gcd(&other.value);
        ExtendedGcd {
            gcd: Self {
                header: self.header,
                value: extended.gcd,
            },
            x: Self {
                header: self.header,
                value: extended.x,
            },
            y: Self {
                header: self.header,
                value: extended.y,
            },
        }
    }
}
impl Add for BigInt {
    type Output = BigInt;

    fn add(mut self, other: Self) -> Self::Output {
        self.value += other.value;
        self
    }
}
impl Sub for BigInt {
    type Output = BigInt;

    fn sub(mut self, other: Self) -> Self::Output {
        self.value -= other.value;
        self
    }
}
impl Mul for BigInt {
    type Output = BigInt;

    fn mul(mut self, other: Self) -> Self::Output {
        self.value *= other.value;
        self
    }
}
impl Div for BigInt {
    type Output = BigInt;

    fn div(mut self, other: Self) -> Self::Output {
        self.value /= other.value;
        self
    }
}
impl Rem for BigInt {
    type Output = BigInt;

    fn rem(mut self, other: Self) -> Self::Output {
        self.value %= other.value;
        self
    }
}
impl Pow<u64> for BigInt {
    type Output = BigInt;

    fn pow(self, rhs: u64) -> Self::Output {
        let value = self.value.pow(rhs);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl Shl<u32> for BigInt {
    type Output = BigInt;

    fn shl(mut self, rhs: u32) -> BigInt {
        self.value <<= rhs;
        self
    }
}
impl Shl<i64> for BigInt {
    type Output = BigInt;

    fn shl(mut self, rhs: i64) -> BigInt {
        self.value <<= rhs;
        self
    }
}
impl Shr<u32> for BigInt {
    type Output = BigInt;

    fn shr(mut self, rhs: u32) -> BigInt {
        self.value >>= rhs;
        self
    }
}
impl Shr<i64> for BigInt {
    type Output = BigInt;

    fn shr(mut self, rhs: i64) -> BigInt {
        self.value >>= rhs;
        self
    }
}
impl BitAnd for BigInt {
    type Output = BigInt;

    fn bitand(mut self, other: Self) -> Self::Output {
        self.value &= other.value;
        self
    }
}
impl BitOr for BigInt {
    type Output = BigInt;

    fn bitor(mut self, other: Self) -> Self::Output {
        self.value |= other.value;
        self
    }
}
impl BitXor for BigInt {
    type Output = BigInt;

    fn bitxor(mut self, other: Self) -> Self::Output {
        self.value ^= other.value;
        self
    }
}
impl Neg for BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        let value = -self.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl Not for BigInt {
    type Output = BigInt;

    fn not(self) -> Self::Output {
        let value = !self.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Add<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn add(mut self, other: &'a BigInt) -> Self::Output {
        self.value += &other.value;
        self
    }
}
impl<'a> Sub<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn sub(mut self, other: &'a BigInt) -> Self::Output {
        self.value -= &other.value;
        self
    }
}
impl<'a> Mul<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn mul(mut self, other: &'a BigInt) -> Self::Output {
        self.value *= &other.value;
        self
    }
}
impl<'a> Div<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn div(mut self, other: &'a BigInt) -> Self::Output {
        self.value /= &other.value;
        self
    }
}
impl<'a> Rem<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn rem(mut self, other: &'a BigInt) -> Self::Output {
        self.value %= &other.value;
        self
    }
}
impl<'a> BitAnd<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn bitand(mut self, other: &'a BigInt) -> Self::Output {
        self.value &= &other.value;
        self
    }
}
impl<'a> BitOr<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn bitor(mut self, other: &'a BigInt) -> Self::Output {
        self.value |= &other.value;
        self
    }
}
impl<'a> BitXor<&'a BigInt> for BigInt {
    type Output = BigInt;

    fn bitxor(mut self, other: &'a BigInt) -> Self::Output {
        self.value ^= &other.value;
        self
    }
}
impl<'a> Neg for &'a BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        let value = -self.value.clone();
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Not for &'a BigInt {
    type Output = BigInt;

    fn not(self) -> Self::Output {
        let value = !self.value.clone();
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Add<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn add(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value += other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Sub<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn sub(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value -= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Mul<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn mul(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value *= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Div<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn div(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value /= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> Rem<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn rem(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value %= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> BitAnd<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitand(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value &= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> BitOr<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitor(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value |= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a> BitXor<BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitxor(self, other: BigInt) -> Self::Output {
        let mut value = self.value.clone();
        value ^= other.value;
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> Add<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn add(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).add(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> Sub<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn sub(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).sub(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> Mul<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn mul(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).mul(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> Div<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn div(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).div(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> Rem<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn rem(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).rem(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> BitAnd<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitand(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).bitand(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> BitOr<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitor(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).bitor(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl<'a, 'b> BitXor<&'b BigInt> for &'a BigInt {
    type Output = BigInt;

    fn bitxor(self, other: &'b BigInt) -> Self::Output {
        let value = (&self.value).bitxor(&other.value);
        BigInt {
            header: self.header,
            value,
        }
    }
}
impl Add<i64> for BigInt {
    type Output = BigInt;

    fn add(self, other: i64) -> Self::Output {
        let value = self.value.add(other);
        Self {
            header: self.header,
            value,
        }
    }
}
impl Sub<i64> for BigInt {
    type Output = BigInt;

    fn sub(self, other: i64) -> Self::Output {
        let value = self.value.sub(other);
        Self {
            header: self.header,
            value,
        }
    }
}
impl Mul<i64> for BigInt {
    type Output = BigInt;

    fn mul(self, other: i64) -> Self::Output {
        let value = self.value.mul(other);
        Self {
            header: self.header,
            value,
        }
    }
}
impl Div<i64> for BigInt {
    type Output = BigInt;

    fn div(self, other: i64) -> Self::Output {
        let value = self.value.div(other);
        Self {
            header: self.header,
            value,
        }
    }
}
impl Rem<i64> for BigInt {
    type Output = BigInt;

    fn rem(self, other: i64) -> Self::Output {
        let value = self.value.rem(other);
        Self {
            header: self.header,
            value,
        }
    }
}
impl BitAnd<i64> for BigInt {
    type Output = BigInt;

    fn bitand(mut self, other: i64) -> Self::Output {
        self.value &= firefly_number::BigInt::from(other);
        self
    }
}
impl BitOr<i64> for BigInt {
    type Output = BigInt;

    fn bitor(mut self, other: i64) -> Self::Output {
        self.value |= firefly_number::BigInt::from(other);
        self
    }
}
impl BitXor<i64> for BigInt {
    type Output = BigInt;

    fn bitxor(mut self, other: i64) -> Self::Output {
        self.value ^= firefly_number::BigInt::from(other);
        self
    }
}
impl Zero for BigInt {
    #[inline]
    fn zero() -> Self {
        Self::new(firefly_number::BigInt::zero())
    }

    #[inline]
    fn set_zero(&mut self) {
        self.value.set_zero();
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}
impl One for BigInt {
    #[inline]
    fn one() -> Self {
        Self::new(firefly_number::BigInt::one())
    }

    #[inline]
    fn set_one(&mut self) {
        self.value.set_one()
    }

    #[inline]
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self.value.is_one()
    }
}
impl ToPrimitive for BigInt {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.value.to_i128()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }

    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.value.to_u128()
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.value.to_f32()
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        self.value.to_f64()
    }
}
