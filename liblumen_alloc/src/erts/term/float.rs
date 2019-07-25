use core::alloc::AllocErr;
use core::cmp::Ordering;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display};
use core::hash::{self, Hash};
use core::ops::*;
use core::ptr;

use num_bigint::{BigInt, Sign};

use crate::borrow::CloneToProcess;

use super::{AsTerm, HeapAlloc, Term};
use super::{BigInteger, SmallInteger};
use crate::erts::term::{TypeError, TypedTerm};

/// A machine-width float, but stored alongside a header value used to identify it in memory
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Float {
    header: Term,
    pub(crate) value: f64,
}
impl Float {
    pub const INTEGRAL_MIN: f64 = -9007199254740992.0;
    pub const INTEGRAL_MAX: f64 = 9007199254740992.0;

    #[cfg(target_pointer_width = "32")]
    const ARITYVAL: usize = 2;
    #[cfg(target_pointer_width = "64")]
    const ARITYVAL: usize = 1;

    pub fn clamp_inclusive_range(overflowing_range: RangeInclusive<f64>) -> RangeInclusive<f64> {
        Self::clamp_value(overflowing_range.start().clone())
            ..=Self::clamp_value(overflowing_range.end().clone())
    }

    #[inline]
    pub fn new(value: f64) -> Self {
        Self {
            header: Term::make_header(Self::ARITYVAL, Term::FLAG_FLOAT),
            value: Self::clamp_value(value),
        }
    }

    #[inline]
    pub fn from_raw(term: *mut Float) -> Self {
        unsafe { *term }
    }

    fn clamp_value(overflowing: f64) -> f64 {
        if overflowing == core::f64::NEG_INFINITY {
            core::f64::MIN
        } else if overflowing == core::f64::INFINITY {
            core::f64::MAX
        } else {
            overflowing
        }
    }
}
unsafe impl AsTerm for Float {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}
impl CloneToProcess for Float {
    #[inline]
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        unsafe {
            let ptr = heap.alloc(self.size_in_words())?.as_ptr() as *mut Self;
            ptr::copy_nonoverlapping(self as *const Self, ptr, 1);
            Ok(Term::make_boxed(ptr))
        }
    }
}
impl From<SmallInteger> for Float {
    #[inline]
    fn from(n: SmallInteger) -> Self {
        Self::new(n.0 as f64)
    }
}
impl From<f64> for Float {
    #[inline]
    fn from(f: f64) -> Self {
        Self::new(f)
    }
}
impl From<f32> for Float {
    #[inline]
    fn from(f: f32) -> Self {
        Self::new(f.into())
    }
}
impl Into<f64> for Float {
    #[inline]
    fn into(self) -> f64 {
        self.value
    }
}
impl Eq for Float {}
impl PartialEq for Float {
    #[inline]
    fn eq(&self, other: &Float) -> bool {
        self.value == other.value
    }
}
impl PartialEq<f64> for Float {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        self.value == *other
    }
}
impl PartialEq<f32> for Float {
    #[inline]
    fn eq(&self, other: &f32) -> bool {
        self.value == (*other).into()
    }
}
impl PartialEq<SmallInteger> for Float {
    #[inline]
    fn eq(&self, other: &SmallInteger) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Equal) => true,
            _ => false,
        }
    }
}
impl PartialEq<BigInteger> for Float {
    #[inline]
    fn eq(&self, other: &BigInteger) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Equal) => true,
            _ => false,
        }
    }
}
impl PartialOrd for Float {
    #[inline]
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl PartialOrd<f64> for Float {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.value.partial_cmp(other)
    }
}
impl PartialOrd<f32> for Float {
    #[inline]
    fn partial_cmp(&self, other: &f32) -> Option<Ordering> {
        let n: f64 = (*other).into();
        self.value.partial_cmp(&n)
    }
}
impl PartialOrd<SmallInteger> for Float {
    #[inline]
    fn partial_cmp(&self, other: &SmallInteger) -> Option<Ordering> {
        use core::num::FpCategory;

        let is_negative = self.value.is_sign_negative();
        match self.value.classify() {
            FpCategory::Nan => None,
            FpCategory::Subnormal => {
                // The float is less precise, so convert to isize
                let f = self.value as isize;
                Some(f.cmp(&other.0))
            }
            FpCategory::Infinite if is_negative => Some(Ordering::Less),
            FpCategory::Infinite => Some(Ordering::Greater),
            FpCategory::Zero => Some(0.cmp(&other.0)),
            FpCategory::Normal => {
                // Float is higher precision
                let i = other.0 as f64;
                self.value.partial_cmp(&i)
            }
        }
    }
}
impl PartialOrd<BigInteger> for Float {
    #[inline]
    fn partial_cmp(&self, other: &BigInteger) -> Option<Ordering> {
        use core::num::FpCategory;
        use num_traits::Zero;

        let is_negative = self.value.is_sign_negative();
        match self.value.classify() {
            FpCategory::Nan => None,
            FpCategory::Subnormal => {
                // The float is less precise, so convert to isize
                let f = BigInt::from(self.value as isize);
                Some(f.cmp(&other.value))
            }
            FpCategory::Infinite if is_negative => Some(Ordering::Less),
            FpCategory::Infinite => Some(Ordering::Greater),
            FpCategory::Zero => {
                let f: BigInt = Zero::zero();
                Some(f.cmp(&other.value))
            }
            FpCategory::Normal => {
                use num_traits::ToPrimitive;
                // Float is higher precision, try and convert to it,
                // if we fail, then the bigint is larger in either direction,
                // which we must determine based on its sign
                if let Some(i) = other.value.to_isize() {
                    return self.value.partial_cmp(&(i as f64));
                }
                if let Sign::Minus = other.value.sign() {
                    return Some(Ordering::Greater);
                }
                Some(Ordering::Less)
            }
        }
    }
}

impl TryFrom<Term> for Float {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Float {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed) => boxed.to_typed_term().unwrap().try_into(),
            TypedTerm::Float(float) => Ok(float),
            _ => Err(TypeError),
        }
    }
}

impl Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Float")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("value", &self.value)
            .finish()
    }
}
impl Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use Debug format so that decimal point is always included so that it is obvious it is a
        // float and not an integer
        write!(f, "{:?}", self.value)
    }
}
impl Hash for Float {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.value.to_bits().hash(state);
    }
}

macro_rules! float_op_trait_impl {
    ($trait:ty, $fun:ident) => {
        impl $trait for Float {
            type Output = Float;
            #[inline]
            fn $fun(self, rhs: Float) -> Self::Output {
                Self::new(self.value.$fun(rhs.value))
            }
        }
    };
}

float_op_trait_impl!(Add, add);
float_op_trait_impl!(Sub, sub);
float_op_trait_impl!(Mul, mul);
float_op_trait_impl!(Div, div);
float_op_trait_impl!(Rem, rem);

impl Neg for Float {
    type Output = Float;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(self.value.neg())
    }
}
