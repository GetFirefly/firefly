///! The way we represent Erlang terms is somewhat similar to ERTS, but different in a number of material
///! aspects:
///!
///! * We choose a smaller number of terms to represent as immediates, and modify which terms are immediates
///! and what their ranges/representation are:
///!    - Floats are immediate
///!    - Pid/Port are never immediate
///!    - SmallInteger is 51-bits wide
///!    - Pointers to Tuple/Cons can be type checked without dereferencing the pointer
///! * Like ERTS, we special case cons cells for more efficient use of memory, but we use a more flexible scheme
///! for boxed terms in general, allowing us to store any Rust type on a process heap. This scheme comes at a
///! slight increase in memory usage for some terms, but lets us have an open set of types, as we don't have to
///! define an encoding scheme for each type individually.
///!
///! In order to properly represent the breadth of Rust types using thin pointers, we use a special smart pointer
///! type called `GcBox<T>` which makes use of the `ptr_metadata` feature to obtain the pointer metadata for a type
///! and store it alongside the allocated data itself. This allows us to use thin pointers everywhere, but still use
///! dynamically-sized types.
///!
///! # Encoding Scheme
///!
///! We've chosen to use a NaN-boxing encoding scheme for immediates. In short, we can hide useful data in the shadow
///! of floating-point NaNs. As a review, IEEE-764 double-precision floating-point values have the following representation
///! in memory:
///!
///! `SEEEEEEEEEEEQMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM`
///!
///! * `S` is the sign bit
///! * `E` are the exponent bits (11 bits)
///! * `Q+M` are the mantissa bits (52 bits)
///! * `Q` is used in conjunction with a NaN bit pattern to indicate whether the NaN is quiet or signaling (i.e. raises an exception)
///!
///! In Rust, the following are some special-case bit patterns which are relevant:
///!
///! * `0111111111111000000000000000000000000000000000000000000000000000` = NaN (canonical, sets the quiet bit)
///! * `0111111111110000000000000000000000000000000000000000000000000000` = Infinity
///! * `1111111111110000000000000000000000000000000000000000000000000000` = -Infinity
///!
///! Additionally, for NaN, it is only required that the canonical bits are set, the mantissa bits are ignored, which means they
///! can be used.
///!
///! Furthermore, Erlang does not support NaN or the infinities, so those bit patterns can be used as well. In short, we have 52 bits to
///! work with when a float is NaN, which is large enough to store a pointer; and since our pointers must be 8-byte aligned, we have
///! an additional 4 bits for a limited tagging scheme. The only requirement is that we don't permit overlapping bit patterns for two
///! different term types.
///!
///! Now that you understand the background, here is the encoding we use for different immediate values:
///!
///! * `Float` is any bit pattern which represents a valid, non-NaN floating-point value OR canonical NaN (i.e. has the quiet bit set).
///! * `None` uses the canonical NaN pattern (i.e. has the quiet bit set)
///! * `Nil` uses the Infinity bit pattern
///! * `Integer` is any bit pattern with the highest 12-bits set to 1, including the bit pattern for -Infinity, which corresponds to the integer 0.
///! Integers are 52-bits wide as a result.
///!
///! These remaining tags assume that the sign bit is 0, the exponent bits are all 1 (i.e. canonical NaN), and that at least one of the mantissa bits
///! are set, and cannot overlap with canonical NaN.
///!
///! * `GcBox<T>` is indicated when the lowest 4 bits are zero (8-byte alignment), but that at least one of the other mantissa bits is non-zero.
///! Tuple/Cons are never represented using `GcBox<T>`.
///! * `RcBox<T>` is indicated when the lowest 4 bits are set (i.e. all 1s), and at least one of the other mantissa bits is non-zero.
///! * `*const T` (i.e. a pointer to a literal/constant) is the same scheme as `GcBox<T>`, but with the lowest 4 bits equal to 0x01. This allows differentiating
///! between garbage-collected data and literals.
///! * `Atom` is has one general encoding and two special marker values for booleans
///!    * `false` is indicated when all mantissa bits equal to 0x02 (i.e. zero when the atom tag is masked out)
///!    * `true` is indicated when all mantissa bits equal to 0x03 (i.e. one when the atom tag is masked out)
///!    * Any other atom is indicated when the lowest 4 bits is equal to 0x02 and the remaining mantissa bits are non-zero. The value of the atom is a pointer
///!    to AtomData, which is never garbage-collected.
///! * `*mut Cons` is indicated when the lowest 4 bits are equal to 0x04 (or 0x05 for literals). When masked, the remaining value is a pointer to `Cons`
///! * `*mut Tuple` is indicated when the lowest 4 bits are equal to 0x06 (or 0x07 for literals). When masked, the remainig value is a pointer to `usize`. Unlike
///! `Cons`, `Tuple` is a dynamically-sized type, so the pointer given actually points to the metadata (a `usize` value) which can be used to construct
///! the fat pointer `*mut Tuple` via `ptr::from_raw_parts_mut`
///!
///! All non-immediate terms are allocated/referenced via `GcBox<T>`.
///!
use core::fmt;
use core::mem;
use core::ptr::{self, NonNull, Pointee};

use super::{Atom, Cons, Float, Tuple, Value};

use crate::alloc::GcBox;

// Canonical NaN
const NAN: u64 = unsafe { mem::transmute::<f64, u64>(f64::NAN) };
// This value has only set the bit which is used to indicate quiet vs signaling NaN (or NaN vs Infinity in the case of Rust)
const QUIET_BIT: u64 = 1 << 51;
// This value has the bit pattern used for the None term, which reuses the bit pattern for NaN
const NONE: u64 = NAN;
// This value has the bit pattern used for the Nil term, which reuses the bit pattern for Infinity
const INFINITY: u64 = NAN & !QUIET_BIT;
const NIL: u64 = INFINITY;
// This value has only the sign bit set
const SIGN_BIT: u64 = 1 << 63;
// This value has all of the bits set which indicate an integer value. To get the actual integer value, you must mask out
// the other bits and then sign-extend the result based on QUIET_BIT, which is the highest bit an integer value can set
const INTEGER_TAG: u64 = INFINITY | SIGN_BIT;

// This tag when used with pointers, indicates that the pointee is constant, i.e. not garbage-collected
const LITERAL_TAG: u64 = 0x01;
// This tag is only ever set when the value is an atom, but is insufficient on its own to determine which type of atom
const ATOM_TAG: u64 = 0x02;
// This constant is used to represent the boolean false value without any pointer to AtomData
const FALSE: u64 = INFINITY | ATOM_TAG;
// This constant is used to represent the boolean true value without any pointer to AtomData
const TRUE: u64 = INFINITY | ATOM_TAG | 0x01;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a cons pointer
// This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const CONS_TAG: u64 = 0x04;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a tuple pointer
// This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const TUPLE_TAG: u64 = 0x06;

// This mask when applied to a u64 will produce a value that can be compared with the tags above for equality
const TAG_MASK: u64 = 0x07;
// This mask when applied to a u64 will return only the bits which are part of the integer value
// NOTE: The value that is produced requires sign-extension based on whether QUIET_BIT is set
const INT_MASK: u64 = !INTEGER_TAG;
// This mask when applied to a u64 will return a value which can be cast to pointer type and dereferenced
const PTR_MASK: u64 = !(SIGN_BIT | INFINITY | TAG_MASK);

#[derive(Debug)]
pub struct ImmediateOutOfRangeError;

#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct OpaqueTerm(u64);
impl OpaqueTerm {
    /// Represents the constant value used to signal an invalid term/exception
    pub const NONE: Self = Self(NONE);
    /// Represents the constant value associated with the value of an empty list
    pub const NIL: Self = Self(NIL);

    #[inline(always)]
    fn is_nan(self) -> bool {
        self.0 & INFINITY == INFINITY
    }

    /// Returns true if this term is a non-boxed value
    ///
    /// This returns true for floats, small integers, nil, and atoms
    ///
    /// NOTE: This returns false for None, as None is not a valid term value
    pub fn is_immediate(self) -> bool {
        let is_float = !self.is_nan();
        let is_float_or_int = is_float & (self.0 & SIGN_BIT == SIGN_BIT);
        let is_atom_or_nil = (self.0 == NIL) | (self.0 & ATOM_TAG == ATOM_TAG);
        is_float_or_int | is_atom_or_nil
    }

    /// Returns true if this term is a non-null pointer to a boxed term
    ///
    /// This returns false if the value is an immediate or a non-NaN float
    #[inline]
    pub fn is_box(self) -> bool {
        !self.is_immediate()
    }

    /// Returns true if this term is a non-null pointer to a GcBox<T> term
    #[inline]
    pub fn is_gcbox(self) -> bool {
        self.is_box() && self.0 & TAG_MASK == 0
    }

    /// Returns true if this term is a non-null pointer to a RcBox<T> term
    #[inline]
    pub fn is_rcbox(self) -> bool {
        self.is_box() && self.0 & TAG_MASK == TAG_MASK
    }

    /// Returns true if this term is a non-null pointer to a literal term
    #[inline]
    pub fn is_literal(self) -> bool {
        self.is_box() && self.0 & LITERAL_TAG == LITERAL_TAG
    }

    /// Returns true if this term is equivalent to a null pointer, i.e. none
    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == NONE
    }

    /// Returns true only if this term is nil
    #[inline]
    pub fn is_nil(self) -> bool {
        self.0 == NIL
    }

    /// Returns true only if this term is an atom
    #[inline]
    pub fn is_atom(self) -> bool {
        self.is_nan() && (self.0 & SIGN_BIT == 0) && (self.0 & ATOM_TAG == ATOM_TAG)
    }

    /// Returns true only if this term is an immediate integer
    ///
    /// NOTE: This does not return true for big integers
    #[inline]
    pub fn is_integer(self) -> bool {
        self.0 & INTEGER_TAG == INTEGER_TAG
    }

    /// Returns true only if this term is a valid, non-NaN floating-point value
    #[inline]
    pub fn is_float(self) -> bool {
        !self.is_nan()
    }

    /// Extracts the raw pointer to the metadata associated with this term
    ///
    /// # Safety
    ///
    /// This function is entirely unsafe unless you have already previously asserted that the term
    /// is a pointer value. A debug assertion is present to catch improper usages in debug builds,
    /// but it is essential that this is only used in conjunction with proper guards in place.
    #[inline]
    pub unsafe fn as_ptr(self) -> *mut () {
        debug_assert!(self.is_box());

        (self.0 & PTR_MASK) as *mut ()
    }

    /// Extracts the atom value contained in this term.
    pub fn as_atom(self) -> Atom {
        use super::atom::AtomData;

        debug_assert!(self.is_atom());
        match self.0 {
            FALSE => Atom::FALSE,
            TRUE => Atom::TRUE,
            _ => {
                let ptr = (self.0 & PTR_MASK) as *mut AtomData;
                let ptr = unsafe { NonNull::new_unchecked(ptr) };
                ptr.into()
            }
        }
    }

    /// Extracts the integer value contained in this term.
    ///
    /// This function is always memory safe, but if improperly used will cause weird results, so it is important
    /// that you guard usages of this function with proper type checks.
    pub fn as_integer(self) -> i64 {
        const NEG: u64 = INTEGER_TAG | QUIET_BIT;
        // Extract the raw 51-bit signed integer
        let raw = self.0 & INT_MASK;
        // Sign-extend to 64-bits by multiplying the extra bits by 1 if signed, 0 if unsigned
        let sign = ((raw & QUIET_BIT == QUIET_BIT) as u64) * NEG;
        (raw | sign) as i64
    }

    /// Convert this term to a floating-point value without any type checks
    ///
    /// This is always safe as all our non-float values are encoded as NaN
    pub fn as_float(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Returns true if the given i64 value is in the range allowed for immediates
    pub(super) fn is_small_integer(value: i64) -> bool {
        let value = value as u64;
        match value & INTEGER_TAG {
            0 | INTEGER_TAG => true,
            _ => false,
        }
    }
}
impl fmt::Binary for OpaqueTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Binary::fmt(&self.0, f)
    }
}
impl From<bool> for OpaqueTerm {
    #[inline]
    fn from(b: bool) -> Self {
        const BOOL_TAG: u64 = INFINITY | ATOM_TAG;

        Self(b as u64 | BOOL_TAG)
    }
}
impl TryFrom<i64> for OpaqueTerm {
    type Error = ImmediateOutOfRangeError;

    fn try_from(i: i64) -> Result<Self, Self::Error> {
        const NEG: u64 = INTEGER_TAG | QUIET_BIT;
        let u = i as u64;
        match u & NEG {
            // Positive integer
            0 => Ok(Self(u | INTEGER_TAG)),
            // Negative integer
            NEG => Ok(Self(u)),
            // Out of range
            _ => Err(ImmediateOutOfRangeError),
        }
    }
}
impl From<f64> for OpaqueTerm {
    #[inline]
    fn from(f: f64) -> Self {
        Self(f.to_bits())
    }
}
impl From<Float> for OpaqueTerm {
    #[inline]
    fn from(f: Float) -> Self {
        Self(f.to_bits())
    }
}
impl From<Atom> for OpaqueTerm {
    #[inline]
    fn from(a: Atom) -> Self {
        if a.is_boolean() {
            a.as_boolean().into()
        } else {
            Self(a.ptr() as u64 | INFINITY | ATOM_TAG)
        }
    }
}
impl<T: ?Sized> From<GcBox<T>> for OpaqueTerm
where
    crate::gc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn from(boxed: GcBox<T>) -> Self {
        let raw = GcBox::into_raw(boxed) as *const () as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY)
    }
}
impl From<NonNull<Cons>> for OpaqueTerm {
    fn from(ptr: NonNull<Cons>) -> Self {
        let raw = ptr.as_ptr() as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | CONS_TAG)
    }
}
impl From<NonNull<Tuple>> for OpaqueTerm {
    fn from(ptr: NonNull<Tuple>) -> Self {
        let raw = ptr.as_ptr() as *const () as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | TUPLE_TAG)
    }
}
impl From<Value> for OpaqueTerm {
    fn from(value: Value) -> Self {
        match value {
            Value::None => Self::NONE,
            Value::Nil => Self::NIL,
            Value::Bool(b) => b.into(),
            Value::Atom(a) => a.into(),
            Value::Float(f) => f.into(),
            Value::Int(i) => i.try_into().unwrap(),
            Value::BigInt(boxed) => boxed.into(),
            Value::Cons(ptr) => ptr.into(),
            Value::Tuple(ptr) => ptr.into(),
            Value::Map(boxed) => boxed.into(),
            Value::Closure(boxed) => boxed.into(),
            Value::Pid(boxed) => boxed.into(),
            Value::Port(boxed) => boxed.into(),
            Value::Reference(boxed) => boxed.into(),
            Value::Binary(boxed) => boxed.into(),
        }
    }
}
impl Into<Value> for OpaqueTerm {
    fn into(self) -> Value {
        const TUPLE_LITERAL_TAG: u64 = TUPLE_TAG | LITERAL_TAG;
        const CONS_LITERAL_TAG: u64 = CONS_TAG | LITERAL_TAG;

        match self.0 {
            NONE => Value::None,
            NIL => Value::Nil,
            FALSE => Value::Bool(false),
            TRUE => Value::Bool(true),
            i if i & INTEGER_TAG == INTEGER_TAG => Value::Int(self.as_integer()),
            other if self.is_nan() => {
                let tag = (other & TAG_MASK);
                match tag {
                    ATOM_TAG => Value::Atom(self.as_atom()),
                    CONS_TAG | CONS_LITERAL_TAG => {
                        Value::Cons(unsafe { NonNull::new_unchecked(self.as_ptr() as *mut Cons) })
                    }
                    TUPLE_TAG | TUPLE_LITERAL_TAG => {
                        let ptr = unsafe { self.as_ptr() as *mut usize };
                        let metadata = unsafe { *ptr };
                        Value::Tuple(unsafe {
                            NonNull::new_unchecked(ptr::from_raw_parts_mut(ptr.cast(), metadata))
                        })
                    }
                    TAG_MASK => {
                        // This is an RcBox
                        let ptr = unsafe { self.as_ptr() };
                        match unsafe { RcBox::type_id(ptr) } {
                            super::Binary::TYPE_ID => {
                                Value::RcBinary(unsafe { RcBox::from_raw_unchecked(ptr) })
                            }
                            _ => panic!("unknown reference-counted pointer type"),
                        }
                    }
                    0 => {
                        // This is a GcBox
                        let ptr = unsafe { self.as_ptr() };
                        match unsafe { GcBox::type_id(ptr) } {
                            super::BigInteger::TYPE_ID => {
                                Value::BigInt(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Map::TYPE_ID => {
                                Value::Map(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Closure::TYPE_ID => {
                                Value::Closure(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Pid::TYPE_ID => {
                                Value::Pid(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Port::TYPE_ID => {
                                Value::Port(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Reference::TYPE_ID => {
                                Value::Reference(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::Binary::TYPE_ID => {
                                Value::GcBinary(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            super::BitSlice::TYPE_ID => {
                                Value::BitSlice(unsafe { GcBox::from_raw_unchecked(ptr) })
                            }
                            _ => panic!("unknown garbage-collected pointer type"),
                        }
                    }
                    tag => panic!("invalid term tag: {:064b}", tag),
                }
            }
            _ => Value::Float(self.as_float().into()),
        }
    }
}
