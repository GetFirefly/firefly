use core::alloc::Layout;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{self, Hash};
use core::mem;
use core::ops::*;
use core::ptr;

use num_bigint::{BigInt, Sign};
use num_traits::cast::ToPrimitive;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::TermAlloc;
use crate::erts::term::prelude::*;

use super::*;

/// Represents big integer terms.
#[derive(Clone)]
#[repr(C)]
pub struct BigInteger {
    header: Header<BigInteger>,
    pub(crate) value: BigInt,
}
impl_static_header!(BigInteger, Term::HEADER_BIG_INTEGER);
impl BigInteger {
    /// Creates a new BigInteger from a BigInt value
    #[inline]
    pub fn new(value: BigInt) -> Self {
        Self {
            header: Default::default(),
            value,
        }
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let bi = BigInt::parse_bytes(bytes, 10)?;
        Some(Self::new(bi))
    }

    /// Returns the number of one bits in the byte representation
    ///
    /// NOTE: The byte representation of BigInt is compacting, and
    /// as a result, this count cannot be compared against other
    /// counts reliably, for example, `-1` will be represented as a
    /// single byte of `0b11111111`, whereas `-1` of any other primitive
    /// integer type (other than u8) will be `size_of::<T>` bytes of
    /// the same.
    pub fn count_ones(&self) -> u32 {
        self.value
            .to_signed_bytes_be()
            .clone()
            .iter()
            .map(|b| b.count_ones())
            .sum()
    }

    #[inline]
    pub fn sign(&self) -> Sign {
        self.value.sign()
    }

    /// Returns the underlying byte representation, in little-endian order
    #[inline]
    pub fn to_signed_bytes_le(&self) -> Vec<u8> {
        self.value.clone().to_signed_bytes_le()
    }

    /// Returns the underlying byte representation, in big-endian order
    #[inline]
    pub fn to_signed_bytes_be(&self) -> Vec<u8> {
        self.value.clone().to_signed_bytes_be()
    }
}
impl fmt::Debug for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let bytes = self
            .value
            .clone()
            .to_signed_bytes_be()
            .iter()
            .map(|b| format!("{:08b}", b))
            .collect::<Vec<String>>()
            .join("");
        f.debug_struct("BigInteger")
            .field("header", &self.header)
            .field("value", &format_args!("{} ({})", &self.value, &bytes))
            .finish()
    }
}
impl fmt::Display for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl CloneToProcess for BigInteger {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        let layout = Layout::for_value(self);
        let size = layout.size();

        let ptr = unsafe {
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut Self;
            // the `Vec<u32>` `data` in the `BigUInt` that is the `data` in `self` is not copied by
            // `ptr::copy_nonoverlapping`, so clone `self` first to make a disconnected `Vec<u32>`
            // that can't be dropped if `self` is dropped.
            //
            // It would be better if we could allocate directly on the `heap`, but until `BigInt`
            // supports setting the allocator, which only happens when `Vec` does, we can't.
            let heap_clone = self.clone();
            ptr::copy_nonoverlapping(&heap_clone as *const _ as *const u8, ptr as *mut u8, size);

            // Make sure the heap_clone `Vec<u32>` address is not dropped
            mem::forget(heap_clone);

            ptr
        };

        Ok(ptr.into())
    }

    fn size_in_words(&self) -> usize {
        crate::erts::to_word_size(Layout::for_value(self).size())
    }
}
impl From<SmallInteger> for BigInteger {
    #[inline]
    fn from(n: SmallInteger) -> Self {
        Self::new(BigInt::from(n.0 as i64))
    }
}
impl From<u32> for BigInteger {
    #[inline]
    fn from(n: u32) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<i32> for BigInteger {
    #[inline]
    fn from(n: i32) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<usize> for BigInteger {
    #[inline]
    fn from(n: usize) -> Self {
        Self::new(BigInt::from(n as u64))
    }
}
impl From<u64> for BigInteger {
    #[inline]
    fn from(n: u64) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<isize> for BigInteger {
    #[inline]
    fn from(n: isize) -> Self {
        Self::new(BigInt::from(n as i64))
    }
}
impl From<i64> for BigInteger {
    #[inline]
    fn from(n: i64) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<i128> for BigInteger {
    #[inline]
    fn from(n: i128) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<u128> for BigInteger {
    fn from(n: u128) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl Into<BigInt> for BigInteger {
    fn into(self) -> BigInt {
        self.value
    }
}
impl Into<BigInt> for Boxed<BigInteger> {
    fn into(self) -> BigInt {
        self.as_ref().value.clone()
    }
}
impl<'a> Into<&'a BigInt> for &'a BigInteger {
    fn into(self) -> &'a BigInt {
        &self.value
    }
}
impl Into<f64> for Boxed<BigInteger> {
    fn into(self) -> f64 {
        self.as_ref().into()
    }
}
impl Into<f64> for &BigInteger {
    fn into(self) -> f64 {
        let (sign, bytes) = self.value.to_bytes_be();
        let unsigned_f64 = bytes
            .iter()
            .fold(0_f64, |acc, byte| 256.0 * acc + (*byte as f64));

        match sign {
            Sign::Minus => -1.0 * unsigned_f64,
            _ => unsigned_f64,
        }
    }
}
impl TryInto<usize> for Boxed<BigInteger> {
    type Error = TryIntoIntegerError;

    #[inline]
    fn try_into(self) -> Result<usize, Self::Error> {
        self.as_ref().try_into()
    }
}
impl TryInto<usize> for &BigInteger {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<usize, Self::Error> {
        let u: u64 = self.try_into()?;
        u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
    }
}
impl TryInto<u64> for Boxed<BigInteger> {
    type Error = TryIntoIntegerError;

    #[inline]
    fn try_into(self) -> Result<u64, Self::Error> {
        self.as_ref().try_into()
    }
}
impl TryInto<u64> for &BigInteger {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<u64, Self::Error> {
        let big_int: &BigInt = self.into();

        match big_int.to_u64() {
            Some(self_u64) => self_u64
                .try_into()
                .map_err(|_| TryIntoIntegerError::OutOfRange),
            None => Err(TryIntoIntegerError::OutOfRange),
        }
    }
}
impl TryFrom<TypedTerm> for Boxed<BigInteger> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::BigInteger(big) => Ok(big),
            _ => Err(TypeError),
        }
    }
}

impl Eq for BigInteger {}
impl PartialEq for BigInteger {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        self.value.eq(&other.value)
    }
}
impl PartialEq<SmallInteger> for BigInteger {
    #[inline]
    fn eq(&self, other: &SmallInteger) -> bool {
        self.value == BigInt::from(other.0 as i64)
    }
}
impl PartialEq<Float> for BigInteger {
    fn eq(&self, other: &Float) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
impl PartialEq<usize> for BigInteger {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
        self.value.eq(&(*other).into())
    }
}
impl PartialEq<isize> for &BigInteger {
    #[inline]
    fn eq(&self, other: &isize) -> bool {
        self.value.eq(&(*other).into())
    }
}
impl PartialEq<isize> for BigInteger {
    #[inline]
    fn eq(&self, other: &isize) -> bool {
        self.value.eq(&(*other).into())
    }
}
impl PartialEq<f64> for BigInteger {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.value.eq(&(*other as usize).into())
    }
}
impl<T> PartialEq<Boxed<T>> for BigInteger
where
    T: PartialEq<BigInteger>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl Ord for BigInteger {
    #[inline]
    fn cmp(&self, other: &BigInteger) -> Ordering {
        self.value.cmp(&other.value)
    }
}
impl PartialOrd for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &BigInteger) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialOrd<SmallInteger> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}
impl PartialOrd<Float> for BigInteger {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        use Ordering::*;
        use Sign::*;

        let self_big_int = &self.value;
        let other_f64 = other.value();

        let ordering = match self_big_int.sign() {
            Minus => {
                if other_f64 < 0.0 {
                    // fits in small integer so the big other_f64 must be lesser
                    if (SmallInteger::MIN_VALUE as f64) <= other_f64 {
                        Less
                    // big_int can't fit in float, so it must be less than any float
                    } else if (std::f64::MAX_EXP as u64) < self_big_int.bits() {
                        Less
                    // > A float is more precise than an integer until all
                    // > significant figures of the float are to the left of the
                    // > decimal point.
                    } else if Float::INTEGRAL_MIN <= other_f64 {
                        let self_f64: f64 = self.into();

                        f64_cmp_f64(self_f64, other_f64)
                    } else {
                        let other_integral_f64 = other_f64.trunc();
                        let other_big_int = unsafe { integral_f64_to_big_int(other_integral_f64) };

                        match self_big_int.cmp(&other_big_int) {
                            Equal => {
                                let float_fract = other_f64 - other_integral_f64;

                                if float_fract == 0.0 {
                                    Equal
                                } else {
                                    // BigInt Is -N while float is -N.M
                                    Greater
                                }
                            }
                            ordering => ordering,
                        }
                    }
                } else {
                    Less
                }
            }
            // BigInt does not have a zero because zero is a SmallInteger
            NoSign => unreachable!(),
            Plus => {
                if 0.0 < other_f64 {
                    // fits in small integer, so the big integer must be greater
                    if other_f64 <= (SmallInteger::MAX_VALUE as f64) {
                        Greater
                    // big_int can't fit in float, so it must be greater than any float
                    } else if (std::f64::MAX_EXP as u64) < self_big_int.bits() {
                        Greater
                    // > A float is more precise than an integer until all
                    // > significant figures of the float are to the left of the
                    // > decimal point.
                    } else if other_f64 <= Float::INTEGRAL_MAX {
                        let self_f64: f64 = self.into();

                        f64_cmp_f64(self_f64, other_f64)
                    } else {
                        let other_integral_f64 = other_f64.trunc();
                        let other_big_int = unsafe { integral_f64_to_big_int(other_integral_f64) };

                        match self_big_int.cmp(&other_big_int) {
                            Equal => {
                                let other_fract = other_f64 - other_integral_f64;

                                if other_fract == 0.0 {
                                    Equal
                                } else {
                                    // BigInt is N while float is N.M
                                    Less
                                }
                            }
                            ordering => ordering,
                        }
                    }
                } else {
                    Greater
                }
            }
        };

        Some(ordering)
    }
}
impl PartialOrd<usize> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        self.value.partial_cmp(&(*other).into())
    }
}
impl PartialOrd<isize> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &isize) -> Option<Ordering> {
        self.value.partial_cmp(&(*other).into())
    }
}
impl PartialOrd<f64> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.value.partial_cmp(&(*other as usize).into())
    }
}
impl<T> PartialOrd<Boxed<T>> for BigInteger
where
    T: PartialOrd<BigInteger>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl Hash for BigInteger {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

macro_rules! bigint_binop_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for BigInteger {
            type Output = BigInteger;
            #[inline]
            fn $fun(self, rhs: BigInteger) -> Self::Output {
                Self::new(self.value.$fun(rhs.value))
            }
        }
        impl $trait for &BigInteger {
            type Output = BigInteger;
            #[inline]
            fn $fun(self, rhs: &BigInteger) -> Self::Output {
                BigInteger::new(self.value.clone().$fun(rhs.value.clone()))
            }
        }
        impl $trait for Boxed<BigInteger> {
            type Output = BigInteger;
            #[inline]
            fn $fun(self, rhs: Boxed<BigInteger>) -> Self::Output {
                BigInteger::new(self.as_ref().value.clone().$fun(rhs.as_ref().value.clone()))
            }
        }
    };
}
macro_rules! bigint_unaryop_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for BigInteger {
            type Output = BigInteger;
            #[inline]
            fn $fun(self) -> Self::Output {
                Self::new(self.value.$fun())
            }
        }
        impl $trait for &BigInteger {
            type Output = BigInteger;
            #[inline]
            fn $fun(self) -> Self::Output {
                BigInteger::new(self.value.clone().$fun())
            }
        }
    };
}

bigint_binop_trait_impl!(Add, add);
bigint_binop_trait_impl!(Sub, sub);
bigint_binop_trait_impl!(Mul, mul);
bigint_binop_trait_impl!(Div, div);
bigint_binop_trait_impl!(Rem, rem);
bigint_unaryop_trait_impl!(Neg, neg);
bigint_unaryop_trait_impl!(Not, not);

impl Add<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn add(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().add(rhs.value))
    }
}
impl Add<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn add(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.add(rhs.value.clone()))
    }
}

impl Sub<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn sub(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().sub(rhs.value))
    }
}
impl Sub<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn sub(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.sub(rhs.value.clone()))
    }
}

impl Mul<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn mul(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().mul(rhs.value))
    }
}
impl Mul<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn mul(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.mul(rhs.value.clone()))
    }
}

impl Div<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn div(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().div(rhs.value))
    }
}
impl Div<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn div(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.div(rhs.value.clone()))
    }
}

impl BitAnd for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitand(self, rhs: BigInteger) -> Self::Output {
        let lhs = self.value.to_signed_bytes_le();
        let rhs = rhs.value.to_signed_bytes_le();
        let bytes = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.bitand(r))
            .collect::<Vec<u8>>();
        let result = BigInt::from_signed_bytes_le(bytes.as_slice());
        BigInteger::new(result)
    }
}
impl BitAnd for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitand(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitand(rhs.value.clone()))
    }
}
impl BitAnd<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitand(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitand(rhs.value))
    }
}
impl BitAnd<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitand(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.bitand(rhs.value.clone()))
    }
}
impl BitOr for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitor(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.bitor(rhs.value))
    }
}
impl BitOr for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitor(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitor(rhs.value.clone()))
    }
}
impl BitOr<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitor(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitor(rhs.value))
    }
}
impl BitOr<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitor(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.bitor(rhs.value.clone()))
    }
}

impl BitXor for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitxor(self, rhs: BigInteger) -> Self::Output {
        let lhs = self.value.to_signed_bytes_le();
        let rhs = rhs.value.to_signed_bytes_le();
        let bytes = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| l.bitxor(r))
            .collect::<Vec<u8>>();
        let result = BigInt::from_signed_bytes_le(bytes.as_slice());
        BigInteger::new(result)
    }
}
impl BitXor for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitxor(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitxor(rhs.value.clone()))
    }
}
impl BitXor<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitxor(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().bitxor(rhs.value))
    }
}
impl BitXor<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn bitxor(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.bitxor(rhs.value.clone()))
    }
}

impl Rem<BigInteger> for &BigInteger {
    type Output = BigInteger;
    #[inline]
    fn rem(self, rhs: BigInteger) -> Self::Output {
        BigInteger::new(self.value.clone().rem(rhs.value))
    }
}
impl Rem<&BigInteger> for BigInteger {
    type Output = BigInteger;
    #[inline]
    fn rem(self, rhs: &BigInteger) -> Self::Output {
        BigInteger::new(self.value.rem(rhs.value.clone()))
    }
}

impl Shl<usize> for BigInteger {
    type Output = BigInteger;

    fn shl(self, rhs: usize) -> Self::Output {
        BigInteger::new(self.value.shl(rhs))
    }
}
impl Shl<usize> for &BigInteger {
    type Output = BigInteger;

    fn shl(self, rhs: usize) -> Self::Output {
        BigInteger::new(self.value.clone().shl(rhs))
    }
}
impl Shr<usize> for BigInteger {
    type Output = BigInteger;

    fn shr(self, rhs: usize) -> Self::Output {
        BigInteger::new(self.value.shr(rhs))
    }
}
impl Shr<usize> for &BigInteger {
    type Output = BigInteger;

    fn shr(self, rhs: usize) -> Self::Output {
        BigInteger::new(self.value.clone().shr(rhs))
    }
}

fn f64_cmp_f64(left: f64, right: f64) -> Ordering {
    match left.partial_cmp(&right) {
        Some(ordering) => ordering,
        // Erlang doesn't support the floats that can't be compared
        None => unreachable!(),
    }
}

unsafe fn integral_f64_to_big_int(integral: f64) -> BigInt {
    let (mantissa, exponent, sign) = num_traits::Float::integer_decode(integral);
    let mantissa_big_int: BigInt = mantissa.into();

    let scaled = if exponent < 0 {
        let right_shift = (-exponent) as usize;

        mantissa_big_int >> right_shift
    } else if exponent == 0 {
        mantissa_big_int
    } else {
        let left_shift = exponent as usize;

        mantissa_big_int << left_shift
    };

    sign * scaled
}
