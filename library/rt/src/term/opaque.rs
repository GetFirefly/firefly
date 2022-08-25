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
///! These remaining tags must use either the Infinity or canonical NaN bit pattern for their high bits, and must be combined with a unique tag
///! in the lowest 4 bits, but no valid value of any type is allowed to overlap with the canonical NaN pattern, so care must be taken when assigning
///! bits. Currently, pointer-like values use the Infinity pattern + tag bits, and null pointers are disallowed. Atoms use canonical NaN.
///!
///! * `GcBox<T>` is indicated when the lowest 4 bits are zero (8-byte alignment), and that at least one of the other mantissa bits is non-zero.
///! Tuple/Cons are never represented using `GcBox<T>`.
///! * `Rc<T>` uses the tag 0x03, and requires that at least one of the other mantissa bits is non-zero. This overlaps with the tag scheme for the
///! atom `true`, but is differentiated by the fact that `true` requires that all mantissa bits other than the tag are zero.
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
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::num::NonZeroU32;
use core::ptr::{self, NonNull, Pointee};

use super::{atoms, Atom, BinaryData, Cons, Float, Term, Tuple};

use liblumen_alloc::gc::{self, GcBox};
use liblumen_alloc::rc::{self, Rc, Weak};
use liblumen_binary::BinaryFlags;

use crate::function::ErlangResult;

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
const FALSE: u64 = NAN | ATOM_TAG;
// This constant is used to represent the boolean true value without any pointer to AtomData
const TRUE: u64 = FALSE | 0x01;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a cons pointer
// This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const CONS_TAG: u64 = 0x04;
const CONS_LITERAL_TAG: u64 = CONS_TAG | LITERAL_TAG;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a tuple pointer
// This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const TUPLE_TAG: u64 = 0x06;
const TUPLE_LITERAL_TAG: u64 = TUPLE_TAG | LITERAL_TAG;
// This tag is used to mark a pointer allocated via Rc<T>
const RC_TAG: u64 = 0x03;

// This mask when applied to a u64 will produce a value that can be compared with the tags above for equality
const TAG_MASK: u64 = 0x07;
// This mask when applied to a u64 will return only the bits which are part of the integer value
// NOTE: The value that is produced requires sign-extension based on whether QUIET_BIT is set
const INT_MASK: u64 = !INTEGER_TAG;
// This mask when applied to a u64 will return a value which can be cast to pointer type and dereferenced
const PTR_MASK: u64 = !(SIGN_BIT | NAN | TAG_MASK);

// This tag indicates a negative integer (i.e. it has our designated sign bit set)
#[cfg(test)]
const NEG_INTEGER_TAG: u64 = INTEGER_TAG | QUIET_BIT;
// This is the largest negative value allowed in an immediate integer
#[cfg(test)]
const MIN_SMALL: i64 = NEG_INTEGER_TAG as i64;
// This is the largest positive value allowed in an immediate integer
#[cfg(test)]
const MAX_SMALL: i64 = (!NEG_INTEGER_TAG) as i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmediateOutOfRangeError;

/// Represents the primary term types that exist in Erlang
///
/// Some types are compositions of these (e.g. list), and some
/// types have subtypes (e.g. integers can be small or big), but
/// for type checking purposes, these are the types we care about.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum TermType {
    Invalid = 0,
    None,
    Nil,
    Bool,
    Atom,
    Int,
    Float,
    Cons,
    Tuple,
    Map,
    Closure,
    Pid,
    Port,
    Reference,
    Binary,
}

/// An opaque term is a 64-bit integer value that represents an encoded term value of any type.
///
/// An opaque term can be decoded into a concrete type by examining the bit pattern of the raw
/// value, as each type has a unique pattern. The concrete value can then be extracted according
/// to the method used for encoding that type.
///
/// Terms can be encoded as either immediate (i.e. the entire value is represented in the opaque
/// term itself), or boxed (i.e. the value of the opaque term is a pointer to a value on the heap).
///
/// Pointer values encoded in a term must always have at least 8-byte alignment on all supported platforms.
/// This should be ensured by specifying the required minimum alignment on all concrete term types we define,
/// but we also add some debug checks to protect against accidentally attempting to encode invalid pointers.
///
/// The set of types given explicit type tags were selected such that the most commonly used types are the
/// cheapest to type check and decode. In general, we believe the most used to be numbers, atoms, lists, and tuples.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct OpaqueTerm(u64);

impl OpaqueTerm {
    /// Represents the constant value used to signal an invalid term/exception
    pub const NONE: Self = Self(NONE);
    /// Represents the constant value associated with the value of an empty list
    pub const NIL: Self = Self(NIL);
    /// Represents the constant value 0 encoded as an integer
    pub const ZERO: Self = Self(INTEGER_TAG);

    /// Returns this opaque term as a raw u64
    #[inline(always)]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// This is a low-level decoding function written in this specific way in order to
    /// maximize the optimizations the compiler can perform from higher-level conversions
    ///
    /// This function returns false if decoding would fail, leaving the provided `Term` pointer
    /// uninitialized. If decoding succeeds, true is returned, and the provided pointer will be
    /// initialized with a valid `Term` value.
    unsafe fn decode(value: OpaqueTerm, term: *mut Term) -> bool {
        match value.0 {
            NONE => term.write(Term::None),
            NIL => term.write(Term::Nil),
            FALSE => term.write(Term::Bool(false)),
            TRUE => term.write(Term::Bool(true)),
            i if i & INTEGER_TAG == INTEGER_TAG => term.write(Term::Int(value.as_integer())),
            other if value.is_nan() => {
                match other & TAG_MASK {
                    ATOM_TAG => term.write(Term::Atom(value.as_atom())),
                    CONS_TAG | CONS_LITERAL_TAG => {
                        term.write(Term::Cons(NonNull::new_unchecked(
                            value.as_ptr() as *mut Cons
                        )));
                    }
                    TUPLE_TAG | TUPLE_LITERAL_TAG => {
                        term.write(Term::Tuple(value.as_tuple_ptr()));
                    }
                    RC_TAG => {
                        let ptr = value.as_ptr();
                        match Weak::<()>::type_id(ptr) {
                            super::BinaryData::TYPE_ID => {
                                let weak: Weak<_> = Weak::from_raw_unchecked(ptr.cast());
                                term.write(Term::RcBinary(weak));
                            }
                            _ => return false,
                        }
                    }
                    LITERAL_TAG => {
                        // Currently, the only possible type which can be flagged as literal without
                        // any other identifying information is constant BinaryData.
                        //
                        // If this ever changes, we will likely need to introduce some kind of header
                        // to distinguish different types, as we are out of tag bits to use
                        let ptr = value.as_ptr();
                        let flags_ptr: *const BinaryFlags = ptr.cast();
                        let size = (&*flags_ptr).size();
                        let ptr = ptr::from_raw_parts::<super::BinaryData>(ptr.cast(), size);
                        term.write(Term::ConstantBinary(&*ptr));
                    }
                    0 => {
                        // This is a GcBox
                        let ptr = value.as_ptr();
                        match GcBox::<()>::type_id(ptr) {
                            super::BigInteger::TYPE_ID => {
                                term.write(Term::BigInt(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::Map::TYPE_ID => {
                                term.write(Term::Map(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::Closure::TYPE_ID => {
                                term.write(Term::Closure(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::Pid::TYPE_ID => {
                                term.write(Term::Pid(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::Port::TYPE_ID => {
                                term.write(Term::Port(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::Reference::TYPE_ID => {
                                term.write(Term::Reference(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::BinaryData::TYPE_ID => {
                                term.write(Term::HeapBinary(GcBox::from_raw_unchecked(ptr)));
                            }
                            super::BitSlice::TYPE_ID => {
                                term.write(Term::RefBinary(GcBox::from_raw_unchecked(ptr)));
                            }
                            _ => return false,
                        }
                    }
                    _tag => return false,
                }
            }
            _ => term.write(Term::Float(value.as_float().into())),
        }

        true
    }

    /// Follows the same rules as `decode`, but simply returns the detected term type
    #[inline]
    pub fn r#typeof(self) -> TermType {
        match self.0 {
            NONE => TermType::None,
            NIL => TermType::Nil,
            FALSE | TRUE => TermType::Bool,
            i if i & INTEGER_TAG == INTEGER_TAG => TermType::Int,
            _other if !self.is_nan() => TermType::Float,
            other => {
                match other & TAG_MASK {
                    ATOM_TAG => TermType::Atom,
                    CONS_TAG | CONS_LITERAL_TAG => TermType::Cons,
                    TUPLE_TAG | TUPLE_LITERAL_TAG => TermType::Tuple,
                    RC_TAG => {
                        let ptr = unsafe { self.as_ptr() };
                        match unsafe { Weak::<()>::type_id(ptr) } {
                            super::BinaryData::TYPE_ID => TermType::Binary,
                            _ => TermType::Invalid,
                        }
                    }
                    LITERAL_TAG => {
                        // Currently, the only possible type which can be flagged as literal without
                        // any other identifying information is constant BinaryData.
                        TermType::Binary
                    }
                    0 => {
                        let ptr = unsafe { self.as_ptr() };
                        match unsafe { GcBox::<()>::type_id(ptr) } {
                            super::BigInteger::TYPE_ID => TermType::Int,
                            super::Map::TYPE_ID => TermType::Map,
                            super::Closure::TYPE_ID => TermType::Closure,
                            super::Pid::TYPE_ID => TermType::Pid,
                            super::Port::TYPE_ID => TermType::Port,
                            super::Reference::TYPE_ID => TermType::Reference,
                            super::BinaryData::TYPE_ID | super::BitSlice::TYPE_ID => {
                                TermType::Binary
                            }
                            _ => TermType::Invalid,
                        }
                    }
                    _invalid => TermType::Invalid,
                }
            }
        }
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        self.0 & INFINITY == INFINITY
    }

    /// Returns true if this term is a non-boxed value
    ///
    /// This returns true for floats, small integers, nil, and atoms
    ///
    /// NOTE: This returns false for None, as None is not a valid term value
    #[inline(always)]
    pub fn is_immediate(self) -> bool {
        !self.is_none() && !self.is_box()
    }

    /// Returns true if this term is a non-null pointer to a boxed term
    ///
    /// This returns false if the value is anything but a pointer to:
    ///
    /// * cons cell
    /// * tuple
    /// * binary literal
    /// * rc-allocated term
    /// * gcbox-allocated term
    #[inline]
    pub fn is_box(self) -> bool {
        // The tag bits uniquely identify all boxed types, but only in conjunction with the hi tag
        match self.0 & TAG_MASK {
            // The only special case is the tag for GcBox overlaps with Nil, but GcBox pointers cannot
            // be null, so that's how we distinguish the two
            0 => self.0 != NIL && self.0 & (NAN | SIGN_BIT) == INFINITY,
            LITERAL_TAG | RC_TAG | CONS_TAG | CONS_LITERAL_TAG | TUPLE_TAG | TUPLE_LITERAL_TAG => {
                // All pointer types use Infinity for their hi tag
                self.0 & (NAN | SIGN_BIT) == INFINITY
            }
            _ => false,
        }
    }

    /// Returns true if this term is a non-null pointer to a GcBox<T> term
    #[inline]
    pub fn is_gcbox(self) -> bool {
        self.0 & (NAN | SIGN_BIT | TAG_MASK) == INFINITY && self.0 != NIL
    }

    /// Returns true if this term is a non-null pointer to a Rc<T> term
    #[inline]
    pub fn is_rc(self) -> bool {
        self.0 & (NAN | SIGN_BIT | TAG_MASK) == (INFINITY | RC_TAG)
    }

    /// Returns true if this term is a non-null pointer to a literal term
    #[inline]
    pub fn is_literal(self) -> bool {
        self.0 & (NAN | SIGN_BIT | TAG_MASK) == (INFINITY | LITERAL_TAG)
    }

    /// Returns true if this term is the None value
    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == NONE
    }

    /// Returns true only if this term is nil
    #[inline(always)]
    pub fn is_nil(self) -> bool {
        self.0 == NIL
    }

    /// Returns true only if this term is an atom
    #[inline]
    pub fn is_atom(self) -> bool {
        const IS_ATOM: u64 = NAN | ATOM_TAG;
        const IS_TRUE: u64 = TRUE;
        match self.0 & (NAN | SIGN_BIT | TAG_MASK) {
            IS_ATOM | IS_TRUE => true,
            _ => false,
        }
    }

    /// Returns true only if this term is an immediate integer
    ///
    /// NOTE: This does not return true for big integers
    #[inline(always)]
    pub fn is_integer(self) -> bool {
        self.0 & INTEGER_TAG == INTEGER_TAG
    }

    /// Returns true only if this term is a valid, non-NaN floating-point value
    #[inline(always)]
    pub fn is_float(self) -> bool {
        !self.is_nan()
    }

    /// Returns true if this term is any type of integer or float
    #[inline]
    pub fn is_number(self) -> bool {
        match self.r#typeof() {
            TermType::Int | TermType::Float => true,
            _ => false,
        }
    }

    /// Returns true if this term is a cons cell pointer
    #[inline]
    pub fn is_nonempty_list(self) -> bool {
        const IS_CONS: u64 = INFINITY | CONS_TAG;
        const IS_CONS_LITERAL: u64 = INFINITY | CONS_LITERAL_TAG;

        match self.0 & (NAN | SIGN_BIT | TAG_MASK) {
            IS_CONS | IS_CONS_LITERAL => true,
            _ => false,
        }
    }

    /// Returns true if this term is nil or a cons cell pointer
    #[inline]
    pub fn is_list(self) -> bool {
        const IS_CONS: u64 = INFINITY | CONS_TAG;
        const IS_CONS_LITERAL: u64 = INFINITY | CONS_LITERAL_TAG;

        match self.0 & (NAN | SIGN_BIT | TAG_MASK) {
            IS_CONS | IS_CONS_LITERAL => true,
            _ => self.0 == NIL,
        }
    }

    /// Returns true if this term is a tuple pointer
    #[inline]
    pub fn is_tuple(self, arity: Option<NonZeroU32>) -> bool {
        match self.tuple_size() {
            ErlangResult::Ok(n) => match arity {
                None => true,
                Some(arity) => arity.get() == n,
            },
            ErlangResult::Err(_) => false,
        }
    }

    /// A combined tuple type test with fetching the arity, optimized for a specific pattern
    /// produced by the compiler
    pub fn tuple_size(self) -> ErlangResult<u32, ()> {
        const IS_TUPLE: u64 = INFINITY | TUPLE_TAG;
        const IS_TUPLE_LITERAL: u64 = INFINITY | TUPLE_LITERAL_TAG;

        match self.0 & (NAN | SIGN_BIT | TAG_MASK) {
            IS_TUPLE | IS_TUPLE_LITERAL => unsafe {
                let ptr = self.as_ptr();
                let meta_ptr: *const usize = ptr.sub(mem::size_of::<usize>()).cast();
                ErlangResult::Ok((*meta_ptr) as u32)
            },
            _ => ErlangResult::Err(()),
        }
    }

    /// Like `erlang:size/1`, but returns the dynamic size of the given term, or 0 if it is not an unsized type
    ///
    /// For tuples, this is the number of elements in the tuple.
    /// For closures, it is the number of elements in the closure environment.
    /// FOr binaries/bitstrings, it is the size in bytes.
    pub fn size(self) -> usize {
        use liblumen_binary::Bitstring;
        match self.into() {
            Term::Tuple(tup) => unsafe { tup.as_ref().len() },
            Term::Closure(fun) => fun.env_size(),
            Term::HeapBinary(bin) => bin.len(),
            Term::RcBinary(bin) => bin.len(),
            Term::RefBinary(slice) => slice.byte_size(),
            Term::ConstantBinary(bin) => bin.len(),
            _ => 0,
        }
    }

    /// Extracts the raw pointer to the metadata associated with this term
    ///
    /// # Safety
    ///
    /// This function is entirely unsafe unless you have already previously asserted that the term
    /// is a pointer value. A debug assertion is present to catch improper usages in debug builds,
    /// but it is essential that this is only used in conjunction with proper guards in place.
    #[inline]
    pub unsafe fn as_ptr(self) -> *mut u8 {
        debug_assert!(self.is_box());

        (self.0 & PTR_MASK) as *mut u8
    }

    /// Extracts a NonNull<Tuple> from this term
    ///
    /// # Safety
    ///
    /// Callers must ensure this opaque term is actually a tuple pointer before calling this.
    unsafe fn as_tuple_ptr(self) -> NonNull<Tuple> {
        // A tuple pointer is a pointer to the first element, but it is preceded by
        // a usize value containing the metadata (i.e. size) for the tuple. To get a
        // fat pointer, we must first access the metadata, then construct the pointer using
        // that metadata
        let ptr = self.as_ptr();
        let meta_ptr: *const usize = ptr.sub(mem::size_of::<usize>()).cast();
        let metadata = *meta_ptr;
        NonNull::new_unchecked(ptr::from_raw_parts_mut(ptr.cast(), metadata))
    }

    /// Extracts the atom value contained in this term.
    pub fn as_atom(self) -> Atom {
        use super::atom::AtomData;

        debug_assert!(self.is_atom());
        match self.0 {
            FALSE => atoms::False,
            TRUE => atoms::True,
            _ => {
                let ptr = (self.0 & PTR_MASK) as *mut AtomData;
                debug_assert_ne!(ptr, 0usize as *mut AtomData);
                debug_assert_ne!(ptr, 1usize as *mut AtomData);
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

    /// This function can be called when cloning a term that might be reference-counted
    pub fn maybe_increment_refcount(&self) {
        if self.is_rc() {
            // We don't need to cast to a concrete type, as it does not matter for this operation
            let boxed = ManuallyDrop::new(unsafe { Rc::<()>::from_raw_unchecked(self.as_ptr()) });
            Rc::increment_strong_count(&*boxed);
        }
    }

    /// This function can be called when dropping a term that might be reference-counted
    pub fn maybe_decrement_refcount(&self) {
        if !self.is_rc() {
            return;
        }

        let ptr = unsafe { self.as_ptr() };
        match unsafe { Rc::<()>::type_id(ptr) } {
            super::BinaryData::TYPE_ID => {
                let _: Rc<super::BinaryData> = unsafe { Rc::from_raw_unchecked(ptr) };
            }
            _ => {
                todo!("should implement a smarter rc container so we can call destructors opaquely")
            }
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
        Self(b as u64 | FALSE)
    }
}
impl From<char> for OpaqueTerm {
    #[inline]
    fn from(c: char) -> Self {
        Self((c as u64) | INTEGER_TAG)
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
        Self(unsafe { a.as_ptr() as u64 | FALSE })
    }
}
impl<T: ?Sized> From<GcBox<T>> for OpaqueTerm
where
    gc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
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
impl<T: ?Sized> From<Rc<T>> for OpaqueTerm
where
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn from(boxed: Rc<T>) -> Self {
        let raw = Rc::into_raw(boxed) as *const () as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | RC_TAG)
    }
}
impl<T: ?Sized> From<Weak<T>> for OpaqueTerm
where
    rc::PtrMetadata: From<<T as Pointee>::Metadata> + TryInto<<T as Pointee>::Metadata>,
{
    fn from(weak: Weak<T>) -> Self {
        let raw = Weak::into_raw(weak) as *const () as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | RC_TAG)
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
impl From<&'static BinaryData> for OpaqueTerm {
    fn from(data: &'static BinaryData) -> Self {
        let raw = data as *const _ as *const () as u64;
        debug_assert!(
            raw & INFINITY == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | LITERAL_TAG)
    }
}
impl From<Term> for OpaqueTerm {
    fn from(term: Term) -> Self {
        match term {
            Term::None => Self::NONE,
            Term::Nil => Self::NIL,
            Term::Bool(b) => b.into(),
            Term::Atom(a) => a.into(),
            Term::Int(i) => i.try_into().unwrap(),
            Term::BigInt(boxed) => boxed.into(),
            Term::Float(f) => f.into(),
            Term::Cons(ptr) => ptr.into(),
            Term::Tuple(ptr) => ptr.into(),
            Term::Map(boxed) => boxed.into(),
            Term::Closure(boxed) => boxed.into(),
            Term::Pid(boxed) => boxed.into(),
            Term::Port(boxed) => boxed.into(),
            Term::Reference(boxed) => boxed.into(),
            Term::HeapBinary(boxed) => boxed.into(),
            Term::RcBinary(weak) => weak.into(),
            Term::RefBinary(boxed) => boxed.into(),
            Term::ConstantBinary(bytes) => bytes.into(),
        }
    }
}
impl Into<Term> for OpaqueTerm {
    #[inline]
    fn into(self) -> Term {
        let mut term = MaybeUninit::uninit();
        unsafe {
            let valid = Self::decode(self, term.as_mut_ptr());
            debug_assert!(valid, "improperly encoded opaque term: {:064b}", self.0);
            term.assume_init()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_matches::assert_matches;

    use alloc::alloc::Global;
    use alloc::alloc::Layout;
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::mem::MaybeUninit;
    use core::num::NonZeroU32;
    use core::ptr::NonNull;

    use liblumen_arena::DroplessArena;
    use liblumen_binary::{BinaryFlags, Bitstring, Encoding, Selection};

    use crate::term::*;

    use super::*;

    #[derive(Default)]
    struct ConstantPool {
        argv: Vec<&'static BinaryData>,
        arena: DroplessArena,
    }
    impl ConstantPool {
        fn insert(&mut self, bytes: &[u8]) -> &'static BinaryData {
            let data = unsafe { self.alloc(bytes) };
            self.argv.push(data);
            data
        }

        unsafe fn alloc(&mut self, bytes: &[u8]) -> &'static BinaryData {
            // Allocate memory for binary metadata and value
            let size = bytes.len();
            let (layout, value_offset) = Layout::new::<BinaryFlags>()
                .extend(Layout::from_size_align_unchecked(
                    size,
                    mem::align_of::<u8>(),
                ))
                .unwrap();
            let layout = layout.pad_to_align();
            let ptr = self.arena.alloc_raw(layout);

            // Write flags
            let flags_ptr: *mut BinaryFlags = ptr.cast();
            flags_ptr.write(BinaryFlags::new_literal(size, Encoding::detect(bytes)));

            // Write data
            let bytes_ptr: *mut u8 = (flags_ptr as *mut u8).add(value_offset);
            ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_ptr, size);

            // Reify as static reference
            let data_ptr: *const BinaryData = ptr::from_raw_parts(ptr.cast(), size);
            &*data_ptr
        }
    }

    #[test]
    fn opaque_term_decode() {
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(OpaqueTerm::NONE, term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::None);

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(OpaqueTerm::NIL, term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Nil);

        // Float
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(f64::MAX.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Float(f64::MAX.into()));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(f64::MIN.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Float(f64::MIN.into()));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(f64::NAN.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::None);

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode((-1.0f64).into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Float((-1.0).into()));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(0.0f64.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Float(0.0.into()));

        // Integer
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(OpaqueTerm::ZERO, term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Int(0));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode((-1i64).try_into().unwrap(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Int(-1));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(MIN_SMALL.try_into().unwrap(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Int(MIN_SMALL));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(MAX_SMALL.try_into().unwrap(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Int(MAX_SMALL));

        // Atoms
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(atoms::False.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Bool(false));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(atoms::True.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Bool(true));

        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(atoms::Error.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Atom(atoms::Error));

        // Cons
        let mut term = MaybeUninit::zeroed();
        let cons = Box::into_raw(Cons::new(Term::Nil, Term::Nil));
        let ptr = unsafe { NonNull::new_unchecked(cons) };
        assert!(unsafe { OpaqueTerm::decode(ptr.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Cons(ptr));

        // Tuple
        let mut term = MaybeUninit::zeroed();
        let ptr = Tuple::from_slice(&[atoms::Ok.into(), OpaqueTerm::NIL], Global).unwrap();
        assert!(unsafe { OpaqueTerm::decode(ptr.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Tuple(ptr));

        // Map
        let mut term = MaybeUninit::zeroed();
        let mut map = Map::new_in(Global).unwrap();
        map.insert_mut(Term::Int(1), Term::Atom(atoms::True));
        assert!(unsafe { OpaqueTerm::decode(map.into(), term.as_mut_ptr()) });
        let map = unsafe { term.assume_init() };
        assert_matches!(map, Term::Map(_));
        let Term::Map(map) = map else { unreachable!(); };
        assert_eq!(map.get(Term::Int(1)), Some(Term::Atom(atoms::True)));

        // Closure
        let mut term = MaybeUninit::zeroed();
        let fun = erlang_error_1 as *const ();
        let closure = Closure::new_in(atoms::Erlang, atoms::Error, 1, fun, &[], Global).unwrap();
        assert!(unsafe { OpaqueTerm::decode(closure.into(), term.as_mut_ptr()) });
        let closure = unsafe { term.assume_init() };
        assert_matches!(closure, Term::Closure(_));
        let Term::Closure(closure) = closure else { unreachable!() };
        assert_eq!(closure.callee(), fun);

        // Pid
        let mut term = MaybeUninit::zeroed();
        let pid = Pid::new_local(1, 1).unwrap();
        let pid2 = GcBox::new(pid.clone());
        assert!(unsafe { OpaqueTerm::decode(pid2.into(), term.as_mut_ptr()) });
        let pid2 = unsafe { term.assume_init() };
        assert_matches!(pid2, Term::Pid(_));
        assert_eq!(pid2, Term::Pid(GcBox::new(pid)));

        // Port
        let mut term = MaybeUninit::zeroed();
        let port = Port::Local {
            id: unsafe { PortId::from_raw(1) },
        };
        let port2 = GcBox::new(port.clone());
        assert!(unsafe { OpaqueTerm::decode(port2.into(), term.as_mut_ptr()) });
        let port2 = unsafe { term.assume_init() };
        assert_matches!(port2, Term::Port(_));
        assert_eq!(port2, Term::Port(GcBox::new(port)));

        // Reference
        let mut term = MaybeUninit::zeroed();
        let reference = Reference::Local {
            id: ReferenceId::new(1, 1),
        };
        let reference2 = GcBox::new(reference.clone());
        assert!(unsafe { OpaqueTerm::decode(reference2.into(), term.as_mut_ptr()) });
        let reference2 = unsafe { term.assume_init() };
        assert_matches!(reference2, Term::Reference(_));
        assert_eq!(reference2, Term::Reference(GcBox::new(reference)));

        // Binary
        let rc = BinaryData::from_str("testing 1 2 3");
        let weak = Rc::into_weak(rc.clone());
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(weak.into(), term.as_mut_ptr()) });
        let weak = unsafe { term.assume_init() };
        assert_matches!(weak, Term::RcBinary(_));
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(rc.into(), term.as_mut_ptr()) });
        let rc = unsafe { term.assume_init() };
        assert_matches!(rc, Term::RcBinary(_));

        // Constant Binary
        let mut constants = ConstantPool::default();
        let string = "testing 1 2 3";
        let bin = constants.insert(string.as_bytes());
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(bin.into(), term.as_mut_ptr()) });
        let bin = unsafe { term.assume_init() };
        assert_matches!(bin, Term::ConstantBinary(_));
        let Term::ConstantBinary(bin) = bin else { unreachable!(); };
        assert_eq!(bin.as_str(), Some("testing 1 2 3"));

        // Binary Reference
        let bits =
            unsafe { core::mem::transmute::<_, &'static dyn Bitstring>(&bin as &dyn Bitstring) };
        let selection = Selection::from_bitstring(bits);
        let bin = GcBox::new_in(BitSlice::from_selection(bin.into(), selection), Global).unwrap();
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(bin.into(), term.as_mut_ptr()) });
        let bin = unsafe { term.assume_init() };
        assert_matches!(bin, Term::RefBinary(_));
        let Term::RefBinary(bin) = bin else { unreachable!(); };
        assert_eq!(bin.as_str(), Some("testing 1 2 3"));
    }

    #[test]
    fn opaque_term_typeof() {
        use crate::term::{Closure, Pid, Port, Reference};

        assert_eq!(OpaqueTerm::NONE.r#typeof(), TermType::None);
        assert_eq!(OpaqueTerm::NIL.r#typeof(), TermType::Nil);

        // Integer
        assert_eq!(OpaqueTerm::ZERO.r#typeof(), TermType::Int);
        let term: OpaqueTerm = (-1i64).try_into().unwrap();
        assert_eq!(term.r#typeof(), TermType::Int);
        let term: OpaqueTerm = MIN_SMALL.try_into().unwrap();
        assert_eq!(term.r#typeof(), TermType::Int);
        let term: OpaqueTerm = MAX_SMALL.try_into().unwrap();
        assert_eq!(term.r#typeof(), TermType::Int);

        // Float
        let term: OpaqueTerm = f64::MAX.into();
        assert_eq!(term.r#typeof(), TermType::Float);
        let term: OpaqueTerm = f64::MIN.into();
        assert_eq!(term.r#typeof(), TermType::Float);
        let term: OpaqueTerm = f64::NAN.into();

        // Atom
        assert_eq!(term.r#typeof(), TermType::None);
        let term: OpaqueTerm = atoms::False.into();
        assert_eq!(term.r#typeof(), TermType::Bool);
        let term: OpaqueTerm = atoms::True.into();
        assert_eq!(term.r#typeof(), TermType::Bool);
        let term: OpaqueTerm = atoms::Error.into();
        assert_eq!(term.r#typeof(), TermType::Atom);

        // Cons
        let cons = Box::into_raw(Cons::new(Term::Nil, Term::Nil));
        let ptr = unsafe { NonNull::new_unchecked(cons) };
        let term: OpaqueTerm = ptr.into();
        assert_eq!(term.r#typeof(), TermType::Cons);

        // Tuple
        let ptr = Tuple::from_slice(&[atoms::Ok.into(), OpaqueTerm::NIL], Global).unwrap();
        let term: OpaqueTerm = ptr.into();
        assert_eq!(term.r#typeof(), TermType::Tuple);

        // Map
        let map = Map::new_in(Global).unwrap();
        let term: OpaqueTerm = map.into();
        assert_eq!(term.r#typeof(), TermType::Map);

        // Closure
        let closure = Closure::new_in(
            atoms::Erlang,
            atoms::Error,
            1,
            erlang_error_1 as *const (),
            &[],
            Global,
        )
        .unwrap();
        let term: OpaqueTerm = closure.into();
        assert_eq!(term.r#typeof(), TermType::Closure);

        // Pid
        let pid = GcBox::new(Pid::new_local(1, 1).unwrap());
        let term: OpaqueTerm = pid.into();
        assert_eq!(term.r#typeof(), TermType::Pid);

        // Port
        let port = GcBox::new(Port::Local {
            id: unsafe { PortId::from_raw(1) },
        });
        let term: OpaqueTerm = port.into();
        assert_eq!(term.r#typeof(), TermType::Port);

        // Reference
        let reference = GcBox::new(Reference::Local {
            id: ReferenceId::new(1, 1),
        });
        let term: OpaqueTerm = reference.into();
        assert_eq!(term.r#typeof(), TermType::Reference);

        // Binary
        let rc = BinaryData::from_str("testing 1 2 3");
        let weak = Rc::into_weak(rc.clone());
        let rc_term: OpaqueTerm = rc.into();
        let weak_term: OpaqueTerm = weak.into();
        assert_eq!(rc_term.r#typeof(), TermType::Binary);
        assert_eq!(weak_term.r#typeof(), TermType::Binary);

        // Constant Binary
        let mut constants = ConstantPool::default();
        let string = "testing 1 2 3";
        let bin = constants.insert(string.as_bytes());
        let term: OpaqueTerm = bin.into();
        assert_eq!(term.r#typeof(), TermType::Binary);

        // Binary Reference
        let bits =
            unsafe { core::mem::transmute::<_, &'static dyn Bitstring>(&bin as &dyn Bitstring) };
        let selection = Selection::from_bitstring(bits);
        let bin = GcBox::new_in(BitSlice::from_selection(term, selection), Global).unwrap();
        let term: OpaqueTerm = bin.into();
        assert_eq!(term.r#typeof(), TermType::Binary);
    }

    #[test]
    fn opaque_term_none() {
        assert!(OpaqueTerm::NONE.is_nan());
        assert!(OpaqueTerm::NONE.is_none());
        assert!(!OpaqueTerm::NONE.is_nil());
        assert!(!OpaqueTerm::NONE.is_immediate());
        assert!(!OpaqueTerm::NONE.is_box());
        assert!(!OpaqueTerm::NONE.is_gcbox());
        assert!(!OpaqueTerm::NONE.is_rc());
        assert!(!OpaqueTerm::NONE.is_literal());
        assert!(!OpaqueTerm::NONE.is_atom());
        assert!(!OpaqueTerm::NONE.is_integer());
        assert!(!OpaqueTerm::NONE.is_float());
        assert!(!OpaqueTerm::NONE.is_number());
        assert!(!OpaqueTerm::NONE.is_nonempty_list());
        assert!(!OpaqueTerm::NONE.is_list());
        assert!(!OpaqueTerm::NONE.is_tuple(None));
        assert_eq!(OpaqueTerm::NONE.tuple_size(), ErlangResult::Err(()));
    }

    #[test]
    fn opaque_term_float() {
        let max: OpaqueTerm = f64::MAX.into();
        let min: OpaqueTerm = f64::MIN.into();
        let zero: OpaqueTerm = 0.0f64.into();
        let one: OpaqueTerm = 1.0f64.into();
        let neg1: OpaqueTerm = (-1.0f64).into();
        let nan: OpaqueTerm = f64::NAN.into();

        assert_eq!(nan, OpaqueTerm::NONE);

        for float in &[max, min, zero, one, neg1] {
            assert!(!float.is_nan());
            assert!(!float.is_nil());
            assert!(float.is_immediate());
            assert!(!float.is_box());
            assert!(!float.is_gcbox());
            assert!(!float.is_rc());
            assert!(!float.is_literal());
            assert!(!float.is_atom());
            assert!(!float.is_integer());
            assert!(float.is_float());
            assert!(float.is_number());
            assert!(!float.is_nonempty_list());
            assert!(!float.is_list());
            assert!(!float.is_tuple(None));
            assert_eq!(float.tuple_size(), ErlangResult::Err(()));
        }
    }

    #[test]
    fn opaque_term_integer() {
        let max: OpaqueTerm = MAX_SMALL.try_into().unwrap();
        let min: OpaqueTerm = MIN_SMALL.try_into().unwrap();
        let zero: OpaqueTerm = 0i64.try_into().unwrap();
        let one: OpaqueTerm = MAX_SMALL.try_into().unwrap();
        let neg1: OpaqueTerm = (-1i64).try_into().unwrap();
        let invalid: Result<OpaqueTerm, _> = i64::MAX.try_into();

        assert_eq!(zero, OpaqueTerm::ZERO);
        assert_eq!(invalid, Err(ImmediateOutOfRangeError));

        for int in &[max, min, zero, one, neg1] {
            assert!(int.is_nan());
            assert!(!int.is_nil());
            assert!(int.is_immediate());
            assert!(!int.is_box());
            assert!(!int.is_gcbox());
            assert!(!int.is_rc());
            assert!(!int.is_literal());
            assert!(!int.is_atom());
            assert!(int.is_integer());
            assert!(!int.is_float());
            assert!(int.is_number());
            assert!(!int.is_nonempty_list());
            assert!(!int.is_list());
            assert!(!int.is_tuple(None));
            assert_eq!(int.tuple_size(), ErlangResult::Err(()));
        }
    }

    #[test]
    fn opaque_term_atom() {
        let true_bool: OpaqueTerm = true.into();
        let true_atom: OpaqueTerm = atoms::True.into();
        let false_bool: OpaqueTerm = false.into();
        let false_atom: OpaqueTerm = atoms::False.into();
        let error_atom: OpaqueTerm = atoms::Error.into();
        let generated_atom = Atom::str_to_term("opaque_term_atom");
        let new_true_atom = Atom::str_to_term("true");
        let new_false_atom = Atom::str_to_term("false");

        assert_eq!(true_bool, true_atom);
        assert_eq!(true_atom, new_true_atom);
        assert_eq!(false_bool, false_atom);
        assert_eq!(false_atom, new_false_atom);
        assert_ne!(true_bool, false_bool);
        assert_eq!(true_atom.as_atom(), atoms::True);
        assert_eq!(new_true_atom.as_atom(), atoms::True);
        assert_eq!(false_atom.as_atom(), atoms::False);
        assert_eq!(new_false_atom.as_atom(), atoms::False);

        for atom in &[true_atom, false_atom, error_atom, generated_atom] {
            assert!(atom.is_nan());
            assert!(!atom.is_nil());
            assert!(atom.is_immediate());
            assert!(!atom.is_box());
            assert!(!atom.is_gcbox());
            assert!(!atom.is_rc());
            assert!(!atom.is_literal());
            assert!(atom.is_atom());
            assert!(!atom.is_integer());
            assert!(!atom.is_float());
            assert!(!atom.is_number());
            assert!(!atom.is_nonempty_list());
            assert!(!atom.is_list());
            assert!(!atom.is_tuple(None));
            assert_eq!(atom.tuple_size(), ErlangResult::Err(()));
        }
    }

    #[test]
    fn opaque_term_literals() {
        let mut constants = ConstantPool::default();
        let string = "testing 1 2 3";
        let bin = constants.insert(string.as_bytes());
        let term: OpaqueTerm = bin.into();

        assert!(term.is_nan());
        assert!(!term.is_nil());
        assert!(!term.is_immediate());
        assert!(term.is_box());
        assert!(!term.is_gcbox());
        assert!(!term.is_rc());
        assert!(term.is_literal());
        assert!(!term.is_atom());
        assert!(!term.is_integer());
        assert!(!term.is_float());
        assert!(!term.is_number());
        assert!(!term.is_nonempty_list());
        assert!(!term.is_list());
        assert!(!term.is_tuple(None));
        assert_eq!(term.tuple_size(), ErlangResult::Err(()));
    }

    #[test]
    fn opaque_term_nil() {
        assert!(OpaqueTerm::NIL.is_nan());
        assert!(!OpaqueTerm::NIL.is_none());
        assert!(OpaqueTerm::NIL.is_nil());
        assert!(OpaqueTerm::NIL.is_immediate());
        assert!(!OpaqueTerm::NIL.is_box());
        assert!(!OpaqueTerm::NIL.is_gcbox());
        assert!(!OpaqueTerm::NIL.is_rc());
        assert!(!OpaqueTerm::NIL.is_literal());
        assert!(!OpaqueTerm::NIL.is_atom());
        assert!(!OpaqueTerm::NIL.is_integer());
        assert!(!OpaqueTerm::NIL.is_float());
        assert!(!OpaqueTerm::NIL.is_number());
        assert!(!OpaqueTerm::NIL.is_nonempty_list());
        assert!(OpaqueTerm::NIL.is_list());
        assert!(!OpaqueTerm::NIL.is_tuple(None));
        assert_eq!(OpaqueTerm::NIL.tuple_size(), ErlangResult::Err(()));
    }

    #[test]
    fn opaque_term_cons() {
        // A list containing a single empty list, e.g. `[[]]`
        let list = Cons::new(OpaqueTerm::NIL, OpaqueTerm::NIL);
        let list = unsafe { NonNull::new_unchecked(Box::into_raw(list)) };
        let cons: OpaqueTerm = list.into();

        // Ensure we drop the allocation if the test fails
        let drop = unsafe { Box::from_raw(list.as_ptr()) };

        assert!(cons.is_nan());
        assert!(!cons.is_nil());
        assert!(!cons.is_immediate());
        assert!(cons.is_box());
        assert!(!cons.is_gcbox());
        assert!(!cons.is_rc());
        assert!(!cons.is_literal());
        assert!(!cons.is_atom());
        assert!(!cons.is_integer());
        assert!(!cons.is_float());
        assert!(!cons.is_number());
        assert!(cons.is_nonempty_list());
        assert!(cons.is_list());
        assert!(!cons.is_tuple(None));
        assert_eq!(cons.tuple_size(), ErlangResult::Err(()));

        let _ = drop;
    }

    #[test]
    fn opaque_term_tuple() {
        // A list containing a single empty list, e.g. `[[]]`
        let ptr = Tuple::from_slice(
            &[atoms::True.into(), atoms::False.into(), OpaqueTerm::NIL],
            Global,
        )
        .unwrap();
        let tuple = unsafe { &*ptr.as_ptr() };
        assert_eq!(tuple.len(), 3);
        let tuple: OpaqueTerm = ptr.into();

        assert_eq!(ptr.as_ptr().to_raw_parts(), unsafe {
            tuple.as_tuple_ptr().as_ptr().to_raw_parts()
        });
        assert!(tuple.is_nan());
        assert!(!tuple.is_nil());
        assert!(!tuple.is_immediate());
        assert!(tuple.is_box());
        assert!(!tuple.is_gcbox());
        assert!(!tuple.is_rc());
        assert!(!tuple.is_literal());
        assert!(!tuple.is_atom());
        assert!(!tuple.is_integer());
        assert!(!tuple.is_float());
        assert!(!tuple.is_number());
        assert!(!tuple.is_nonempty_list());
        assert!(!tuple.is_list());
        assert_eq!(tuple.tuple_size(), ErlangResult::Ok(3));
        assert!(tuple.is_tuple(None));
        assert!(tuple.is_tuple(NonZeroU32::new(3)));
        assert!(!tuple.is_tuple(NonZeroU32::new(2)));
        assert!(!tuple.is_tuple(NonZeroU32::new(4)));
    }

    #[test]
    fn opaque_term_gcbox() {
        let mut boxed = Map::new_in(Global).unwrap();
        boxed.insert_mut(Term::Int(1), Term::Atom(atoms::True));
        // Save the raw pointer
        let ptr = GcBox::into_raw(boxed);
        let boxed = unsafe { GcBox::from_raw(ptr) };
        let map: OpaqueTerm = boxed.into();

        assert_eq!(ptr as *mut u8, unsafe { map.as_ptr() });
        assert!(map.is_nan());
        assert!(!map.is_nil());
        assert!(!map.is_immediate());
        assert!(map.is_box());
        assert!(map.is_gcbox());
        assert!(!map.is_rc());
        assert!(!map.is_literal());
        assert!(!map.is_atom());
        assert!(!map.is_integer());
        assert!(!map.is_float());
        assert!(!map.is_number());
        assert!(!map.is_nonempty_list());
        assert!(!map.is_list());
        assert!(!map.is_tuple(None));
        assert_eq!(map.tuple_size(), ErlangResult::Err(()));

        let boxed = unsafe { GcBox::from_raw(ptr) };
        unsafe { GcBox::drop_in(boxed, Global) }
    }

    #[test]
    fn opaque_term_rcbox() {
        let boxed = BinaryData::from_str("testing 1 2 3");
        let weak = Rc::into_weak(boxed.clone());
        // Save the raw pointers
        let rc_ptr = Rc::into_raw(boxed);
        let weak_ptr = Weak::into_raw(weak);
        let rc = unsafe { Rc::from_raw(rc_ptr) };
        let weak = unsafe { Weak::from_raw(weak_ptr) };
        let rc_bin: OpaqueTerm = rc.into();
        let weak_bin: OpaqueTerm = weak.into();

        // The pointers should all be to the same object
        assert_eq!(rc_ptr as *mut u8, unsafe { rc_bin.as_ptr() });
        assert_eq!(weak_ptr as *mut u8, unsafe { weak_bin.as_ptr() });
        assert_eq!(unsafe { rc_bin.as_ptr() }, unsafe { weak_bin.as_ptr() });

        for bin in &[rc_bin, weak_bin] {
            assert!(bin.is_nan());
            assert!(!bin.is_nil());
            assert!(!bin.is_immediate());
            assert!(bin.is_box());
            assert!(!bin.is_gcbox());
            assert!(bin.is_rc());
            assert!(!bin.is_literal());
            assert!(!bin.is_atom());
            assert!(!bin.is_integer());
            assert!(!bin.is_float());
            assert!(!bin.is_number());
            assert!(!bin.is_nonempty_list());
            assert!(!bin.is_list());
            assert!(!bin.is_tuple(None));
            assert_eq!(bin.tuple_size(), ErlangResult::Err(()));
        }

        let _ = unsafe { Rc::from_raw(rc_ptr) };
    }

    // Used for closure construction
    fn erlang_error_1(a: OpaqueTerm) -> ErlangResult {
        ErlangResult::Ok(a)
    }
}
