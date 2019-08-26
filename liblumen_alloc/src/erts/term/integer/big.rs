use core::cmp::Ordering;
use core::fmt::{self, Debug, Display};
use core::hash::{self, Hash};
use core::mem;
use core::ops::*;
use core::ptr;

use num_bigint::{BigInt, Sign};
use num_traits::cast::ToPrimitive;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::term::{AsTerm, Boxed, Float, Term, TryIntoIntegerError};
use crate::erts::{to_word_size, HeapAlloc};

use super::*;

/// Represents big integer terms.
#[derive(Clone)]
#[repr(C)]
pub struct BigInteger {
    header: Term,
    pub(crate) value: BigInt,
}
impl BigInteger {
    /// Creates a new BigInteger from a BigInt value
    #[inline]
    pub fn new(value: BigInt) -> Self {
        let header = Term::make_bigint_header(&value);
        Self { header, value }
    }
}
unsafe impl AsTerm for BigInteger {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}
impl CloneToProcess for BigInteger {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        let size = mem::size_of_val(self);
        let size_in_words = to_word_size(size);
        let ptr = unsafe { heap.alloc(size_in_words)?.as_ptr() };

        // the `Vec<u32>` `data` in the `BigUInt` that is the `data` in `self` is not copied by
        // `ptr::copy_nonoverlapping`, so clone `self` first to make a disconnected `Vec<u32>`
        // that can't be dropped if `self` is dropped.
        //
        // It would be better if we could allocate directly on the `heap`, but until `BigInt`
        // supports setting the allocator, which only happens when `Vec` does, we can't.
        let heap_clone = self.clone();

        unsafe {
            ptr::copy_nonoverlapping(&heap_clone as *const _ as *const u8, ptr as *mut u8, size);
        }

        // make sure the heap_clone `Vec<u32>` address is not dropped
        mem::forget(heap_clone);

        let boxed = Term::make_boxed(ptr);

        Ok(boxed)
    }
}
impl From<SmallInteger> for BigInteger {
    #[inline]
    fn from(n: SmallInteger) -> Self {
        Self::new(BigInt::from(n.0 as i64))
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
impl Into<BigInt> for BigInteger {
    fn into(self) -> BigInt {
        self.value
    }
}
impl<'a> Into<&'a BigInt> for &'a BigInteger {
    fn into(self) -> &'a BigInt {
        &self.value
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
impl Into<f64> for Boxed<BigInteger> {
    fn into(self) -> f64 {
        self.as_ref().into()
    }
}
impl Eq for BigInteger {}
impl PartialEq for BigInteger {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        self.value.eq(&other.value)
    }
}
impl PartialEq<Boxed<BigInteger>> for BigInteger {
    fn eq(&self, other: &Boxed<BigInteger>) -> bool {
        self.eq(other.as_ref())
    }
}
impl PartialEq<SmallInteger> for BigInteger {
    #[inline]
    fn eq(&self, other: &SmallInteger) -> bool {
        self.value == BigInt::from(other.0 as i64)
    }
}
impl PartialEq<SmallInteger> for Boxed<BigInteger> {
    fn eq(&self, other: &SmallInteger) -> bool {
        self.as_ref().eq(other)
    }
}
impl PartialEq<Float> for BigInteger {
    fn eq(&self, other: &Float) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
impl PartialEq<Float> for Boxed<BigInteger> {
    fn eq(&self, other: &Float) -> bool {
        self.as_ref().eq(other)
    }
}
impl PartialEq<usize> for BigInteger {
    #[inline]
    fn eq(&self, other: &usize) -> bool {
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
impl PartialOrd<Boxed<BigInteger>> for BigInteger {
    fn partial_cmp(&self, other: &Boxed<BigInteger>) -> Option<Ordering> {
        self.partial_cmp(other.as_ref())
    }
}
impl PartialOrd<SmallInteger> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}
impl PartialOrd<SmallInteger> for Boxed<BigInteger> {
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
    }
}
impl PartialOrd<Float> for BigInteger {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        use Ordering::*;
        use Sign::*;

        let self_big_int = &self.value;
        let other_f64 = other.value;

        let ordering = match self_big_int.sign() {
            Minus => {
                if other_f64 < 0.0 {
                    // fits in small integer so the big other_f64 must be lesser
                    if (SmallInteger::MIN_VALUE as f64) <= other_f64 {
                        Less
                    // big_int can't fit in float, so it must be less than any float
                    } else if (std::f64::MAX_EXP as usize) < self_big_int.bits() {
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
                    } else if (std::f64::MAX_EXP as usize) < self_big_int.bits() {
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
impl PartialOrd<Float> for Boxed<BigInteger> {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        self.as_ref().partial_cmp(other)
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
impl Debug for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BigInteger")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("value", &self.value)
            .finish()
    }
}
impl Display for BigInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
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
    };
}

bigint_binop_trait_impl!(Add, add);
bigint_binop_trait_impl!(Sub, sub);
bigint_binop_trait_impl!(Mul, mul);
bigint_binop_trait_impl!(Div, div);
bigint_binop_trait_impl!(BitAnd, bitand);
bigint_binop_trait_impl!(BitOr, bitor);
bigint_binop_trait_impl!(BitXor, bitxor);
bigint_binop_trait_impl!(Rem, rem);
bigint_unaryop_trait_impl!(Neg, neg);
bigint_unaryop_trait_impl!(Not, not);

impl Shl<usize> for BigInteger {
    type Output = BigInteger;

    fn shl(self, rhs: usize) -> Self {
        BigInteger::new(self.value.shl(rhs))
    }
}
impl Shr<usize> for BigInteger {
    type Output = BigInteger;

    fn shr(self, rhs: usize) -> Self {
        BigInteger::new(self.value.shr(rhs))
    }
}

impl TryInto<u64> for Boxed<BigInteger> {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<u64, Self::Error> {
        let big_int: &BigInt = self.as_ref().into();

        match big_int.to_u64() {
            Some(self_u64) => self_u64
                .try_into()
                .map_err(|_| TryIntoIntegerError::OutOfRange),
            None => Err(TryIntoIntegerError::OutOfRange),
        }
    }
}

impl TryInto<usize> for Boxed<BigInteger> {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<usize, Self::Error> {
        let u: u64 = self.try_into()?;

        u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
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
