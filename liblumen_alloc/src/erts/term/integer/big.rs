use core::alloc::AllocErr;
use core::cmp::Ordering;
use core::fmt::{self, Debug, Display};
use core::hash::{self, Hash};
use core::mem;
use core::ops::*;
use core::ptr;

use num_bigint::{BigInt, Sign};
use num_traits::cast::ToPrimitive;

use crate::borrow::CloneToProcess;
use crate::erts::term::{AsTerm, Boxed, Term, TryIntoIntegerError};
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
        let flag = match value.sign() {
            Sign::NoSign | Sign::Plus => Term::FLAG_POS_BIG_INTEGER,
            Sign::Minus => Term::FLAG_NEG_BIG_INTEGER,
        };
        let arity = to_word_size(mem::size_of_val(&value));
        let header = Term::make_header(arity, flag);
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
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        let size = mem::size_of_val(self);
        let size_in_words = to_word_size(size);
        let ptr = unsafe { heap.alloc(size_in_words)?.as_ptr() };
        unsafe {
            ptr::copy_nonoverlapping(self as *const _ as *const u8, ptr as *mut u8, size);
        }
        Ok(Term::make_boxed(ptr as *mut Self))
    }
}
impl From<SmallInteger> for BigInteger {
    #[inline]
    fn from(n: SmallInteger) -> Self {
        Self::new(BigInt::from(n.0 as i64))
    }
}
impl From<u64> for BigInteger {
    fn from(n: u64) -> Self {
        Self::new(BigInt::from(n))
    }
}
impl From<usize> for BigInteger {
    #[inline]
    fn from(n: usize) -> Self {
        Self::new(BigInt::from(n as u64))
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
impl Into<f64> for Boxed<BigInteger> {
    fn into(self) -> f64 {
        std::dbg!();
        std::dbg!(&self.value);
        let (sign, bytes) = self.value.to_bytes_be();
        std::dbg!();
        let unsigned_f64 = bytes
            .iter()
            .fold(0_f64, |acc, byte| 256.0 * acc + (*byte as f64));
        std::dbg!();

        match sign {
            Sign::Minus => -1.0 * unsigned_f64,
            _ => unsigned_f64,
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
impl PartialOrd<SmallInteger> for BigInteger {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        self.partial_cmp(&other.0)
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
            .field("header", &self.header.as_usize())
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
