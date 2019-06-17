use core::cmp::Ordering;
use core::fmt::{self, Debug, Display};
use core::hash::{self, Hash};
use core::mem;
use core::ops::*;
use core::ptr;

use num_bigint::{BigInt, Sign};

use crate::borrow::CloneToProcess;
use crate::erts::to_word_size;
use crate::erts::{AsTerm, ProcessControlBlock, Term};

use super::*;

/// Represents big integer terms.
///
/// The header field of this struct is assumed to be
/// flagged with FLAG_BIG_INTEGER, and when masked,
/// should contain 0 for unsigned/positive values and
/// 1 for negative values
#[derive(Clone)]
#[repr(C)]
pub struct BigInteger {
    header: usize,
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
        let arity = to_word_size(value.bits() / 8);
        Self { header: arity | flag, value }
    }
}
unsafe impl AsTerm for BigInteger {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::from_raw(self as *const _ as usize | Term::FLAG_BOXED)
    }
}
impl CloneToProcess for BigInteger {
    fn clone_to_process(&self, process: &mut ProcessControlBlock) -> Term {
        let size = mem::size_of_val(self);
        let words = to_word_size(size);
        unsafe {
            let ptr = process.alloc(words).unwrap().as_ptr();
            ptr::copy_nonoverlapping(self as *const _ as *const u8, ptr as *mut u8, size);
            let bi = &*(ptr as *mut Self);
            bi.as_term()
        }
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
impl From<isize> for BigInteger {
    #[inline]
    fn from(n: isize) -> Self {
        Self::new(BigInt::from(n as i64))
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
            .field("header", &self.header)
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
