///! The way we represent Erlang terms is somewhat similar to ERTS, but different in a number
///! of material aspects:
///!
///! * We choose a smaller number of terms to represent as immediates, and modify which terms
///! are immediates and what their ranges/representation are:
///!    - Floats are immediate
///!    - Pid/Port are never immediate
///!    - SmallInteger is 52-bits wide
///!    - Pointers to Tuple/Cons can be type checked without dereferencing the pointer
///! * Like ERTS, we special case cons cells for more efficient use of memory, but we use a
///! more flexible scheme for boxed terms in general, allowing us to store any Rust type on a
///! process heap. This scheme comes at a slight increase in memory usage for some terms, but
///! lets us have an open set of types, as we don't have to define an encoding scheme for each
///! type individually.
///!
///! In order to properly represent the breadth of Rust types using thin pointers, we use a
///! special smart pointer type called `Gc<T>` which makes use of the `ptr_metadata` feature
///! to obtain the pointer metadata for a type and store it alongside the allocated data
///! itself. This allows us to use thin pointers everywhere, but still use dynamically-sized
///! types.
///!
///! # Encoding Scheme
///!
///! We've chosen to use a NaN-boxing encoding scheme for immediates. In short, we can hide
///! useful data in the shadow of floating-point NaNs. As a review, IEEE-764 double-precision
///! floating-point values have the following representation in memory:
///!
///! `SEEEEEEEEEEEQMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM`
///!
///! * `S` is the sign bit
///! * `E` are the exponent bits (11 bits)
///! * `Q+M` are the mantissa bits (52 bits)
///! * `Q` is used in conjunction with a NaN bit pattern to indicate whether the NaN is quiet
///! or signaling (i.e. raises an exception)
///!
///! In Rust, the following are some special-case bit patterns which are relevant:
///!
///! * `0111111111111000000000000000000000000000000000000000000000000000` = NaN
///!   - Canonical
///!   - Sets the quiet bit
///! * `0111111111110000000000000000000000000000000000000000000000000000` = Infinity
///! * `1111111111110000000000000000000000000000000000000000000000000000` = -Infinity
///!
///! Additionally, for NaN, it is only required that the canonical bits are set, the mantissa
///! bits are ignored, which means they can be used.
///!
///! Furthermore, Erlang does not support NaN or the infinities, so those bit patterns can be
///! used as well. This gives us:
///!
///! * Infinity + 51 contiguous bits for value + unique tag
///! * -Infinity + 52 contiguous bits for a primitive i52 integer type
///! * Canonical NaN + 51 contiguous bits for value + unique tag
///!
///! Additionally, we require that:
///!
///! * Pointer addresses are 8-byte aligned, leaving the lowest 3 bits unused for tagging
///! * Reservation of the special NaN, Infinity and -Infinity marker values for internal use
///!
///! # Term Types
///!
///! * None, a singleton invalid value, used for various purposes but not constructible from
///! user code
///! * Nil, a singleton value representing the empty list
///! * Boolean, composed of two singleton values for false and true, also valid atoms
///! * Atom, a pointer to an AtomData struct, unique for each atom; the pointer value is used
///! for cheap equality comparison
///! * Integer, an i52 equivalent immediate integer value
///! * Float, a f64 value, but without support for NaN, or the infinities
///! * Gc/Arc/*const, a pointer to a heap allocated or constant value, depending on pointer
///! type
///! * Catch, a pointer or offset to the instruction to which control will be transferred
///! for a raised exception
///! * Code, a pointer or offset to the instruction to which control will be transferred when
///! returning from a function
///!
///! # Term Encodings
///!
///! * All bit patterns which are non-NaN, non-infinite float values      = Float
///!
///! ## -Infinity
///!
///! The following are the term types which use -Infinity as their tag
///!
///! * `111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` = Integer (-Inf)
///!   - Equivalent to an i52 integer
///!
///! ## Infinity
///!
///! The following are the term types which use Infinity as their tag in the high bits,
///! and use a unique tag in their lowest 3 bits.
///!
///! * `0111111111110000000000000000000000000000000000000000000000000000` = NIL
///! * `0111111111110000000000000000000000000000000000000000000000000010` = FALSE
///! * `0111111111110000000000000000000000000000000000000000000000000011` = TRUE
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx000` = Gc<T>
///!   - A non-null pointer value is required
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx001` = &'static T
///!   - A non-null pointer value is required
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx010` = Atom
///!   - A non-null pointer to AtomData is required
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx011` = Arc<T>
///!   - A non-null pointer value is required
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx100` = Gc<Cons>
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx101` = &'static Cons
///!   - A non-null pointer value is required
///!   - The lowest bit (0x01) is set if the value is a literal
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx110` = Gc<Tuple>
///! * `0111111111110xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx111` = &'static Tuple
///!   - A non-null pointer value is required
///!   - The lowest bit (0x01) is set if the value is a literal
///!
///! ## NaN
///!
///! The following are the term types which use canonical NaN to tag their high bits.
///! These term types are not valid Erlang terms themselves, but are used as markers, or
///! represent various runtime-internal values.
///!
///! In general, the lowest 3 bits are used as a unique tag, but in a couple of cases,
///! additional bits are used to differentiate between overlapping tag bits.
///!
///! * `0111111111111000000000000000000000000000000000000000000000000000` = NONE
///!   - Represents void/()/!
///! * `0111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx001` = Catch
///!   - Encodes an instruction pointer for a catch handler
///!   - Must be non-null
///! * `0111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx010` = Code
///!   - Encodes an instruction pointer for continuations
///!   - May also be used to masquerade a runtime-internal type on the process stack
///!   - Must be non-null
///! * `0111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxTTTT11` = Header
///!   - Represents the start of a heap-allocated term's data
///!   - Has a 4-bit tag for the term type
///!   - Has a 45-bit arity value (typically used for dynamically-sized types)
///! * `0111111111111xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1000` = HOLE
///!   - Represents a region of reclaimed memory immediately following the marker
///!   - Technically overlaps with the tag for NONE, but the value is guaranteed
///!   to be unique due to the use of bit #4 as an additional tag bit.
///!
use alloc::sync::Arc;
use core::fmt;
use core::mem::{self, MaybeUninit};
use core::ptr::NonNull;

use super::{atoms, Atom, Closure, Cons, Float, Term, Tuple};
use super::{BigInt, BinaryData, BitSlice, Map, MatchContext, Pid, Port, Reference};
use super::{Boxable, Header, Tag};

use crate::gc::Gc;

// Canonical NaN
const NAN: u64 = unsafe { mem::transmute::<f64, u64>(f64::NAN) };
// This value has only set the bit which is used to indicate quiet vs signaling NaN (or NaN vs
// Infinity in the case of Rust)
const QUIET_BIT: u64 = 1 << 51;
// This value has the bit pattern used for the None term, which reuses the bit pattern for NaN
const NONE: u64 = NAN;
// This value has the bit pattern used for the Nil term, which reuses the bit pattern for Infinity
const INFINITY: u64 = unsafe { mem::transmute::<f64, u64>(f64::INFINITY) };
const NEG_INFINITY: u64 = unsafe { mem::transmute::<f64, u64>(f64::NEG_INFINITY) };
const NIL: u64 = INFINITY;
// This is an alias for the quiet bit, which is used as the sign bit for integer values
const SIGN_BIT: u64 = QUIET_BIT;
// This value has all of the bits set which indicate an integer value. To get the actual integer
// value, you must mask out the other bits and then sign-extend the result based on QUIET_BIT, which
// is the highest bit an integer value can set
const INTEGER_TAG: u64 = NEG_INFINITY;

// This tag when used with pointers, indicates that the pointee is constant, i.e. not
// garbage-collected
const LITERAL_TAG: u64 = 0x01;
const CATCH_TAG: u64 = 0x01;
// This tag is only ever set when the value is an atom, but is insufficient on its own to determine
// which type of atom
const ATOM_TAG: u64 = 0x02;
const CODE_TAG: u64 = 0x02;
// This constant is used to represent the boolean false value without any pointer to AtomData
const FALSE: u64 = INFINITY | ATOM_TAG;
// This constant is used to represent the boolean true value without any pointer to AtomData
const TRUE: u64 = FALSE | 0x01;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a cons
// pointer This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const CONS_TAG: u64 = 0x04;
const CONS_LITERAL_TAG: u64 = CONS_TAG | LITERAL_TAG;
// This tag represents a unique combination of the lowest 4 bits indicating the value is a tuple
// pointer This tag can be combined with LITERAL_TAG to indicate the pointer is constant
const TUPLE_TAG: u64 = 0x06;
const TUPLE_LITERAL_TAG: u64 = TUPLE_TAG | LITERAL_TAG;
// This tag is used to mark a pointer allocated via Arc<T>
const RC_TAG: u64 = 0x03;
// This tag is used in conjunction with NAN to tag header words for boxed term types on a heap
const HEADER_TAG: u64 = 0x03;
const HOLE_TAG: u64 = 0x08;

// This mask when applied to a u64 distinguishes between NAN/Infinity and -Infinity
const GROUP_MASK: u64 = NEG_INFINITY;
// This mask when applied to a u64 distinguishes between NAN, Infinity and -Infinity
const SUBGROUP_MASK: u64 = NEG_INFINITY | QUIET_BIT;
// This mask when applied to a u64 will produce a value that can be compared with the tags above for
// equality
const TAG_MASK: u64 = 0x07;
// This mask when applied to a u64 will return only the bits which are part of the integer value
// NOTE: The value that is produced requires sign-extension based on whether SIGN_BIT is set
const INT_MASK: u64 = !INTEGER_TAG;
// This mask when applied to a u64 will return a value which can be cast to pointer type and
// dereferenced
const PTR_MASK: u64 = !(NEG_INFINITY | SIGN_BIT | TAG_MASK);
// This extends SUBGROUP_MASK to allow pairing it with special tag bits
const SPECIAL_TAG_MASK: u64 = SUBGROUP_MASK | HEADER_TAG;
// This is the base tag for specials
const SPECIAL_TAG: u64 = NAN;

// This tag indicates a negative integer (i.e. it has our designated sign bit set)
const NEG_INTEGER_TAG: u64 = INTEGER_TAG | SIGN_BIT;
// This constant represents the total number of bits that a small integer can fill
const INT_BITSIZE: usize = INTEGER_TAG.trailing_zeros() as usize;

// This constant has all of the usable unsigned bits of an integer value set
#[cfg(test)]
const UNSIGNED_BITS: u64 = !(NEG_INTEGER_TAG);
// This is the largest negative value allowed in an immediate integer
#[cfg(test)]
const MIN_SMALL: i64 = !UNSIGNED_BITS as i64;
// This is the largest positive value allowed in an immediate integer
#[cfg(test)]
const MAX_SMALL: i64 = UNSIGNED_BITS as i64;

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
    Match,
    Code,
    Catch,
    Header,
    Hole,
}
impl TermType {
    #[inline]
    pub fn is_number(&self) -> bool {
        match self {
            Self::Int | Self::Float => true,
            _ => false,
        }
    }

    pub fn is_special(&self) -> bool {
        match self {
            Self::Code | Self::Catch | Self::Header | Self::Hole | Self::Match => true,
            _ => false,
        }
    }
}
impl PartialOrd for TermType {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TermType {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        // number < atom < reference < fun < port < pid < tuple < map < nil < list < bit string

        if core::mem::discriminant(self) == core::mem::discriminant(other) {
            return Ordering::Equal;
        }

        let lhs_is_number = self.is_number();
        let rhs_is_number = other.is_number();
        if lhs_is_number && rhs_is_number {
            return Ordering::Equal;
        }
        if lhs_is_number {
            return Ordering::Less;
        }
        if rhs_is_number {
            return Ordering::Greater;
        }

        match (self, other) {
            (TermType::Atom, _) => Ordering::Less,
            (_, TermType::Atom) => Ordering::Greater,
            (TermType::Reference, _) => Ordering::Less,
            (_, TermType::Reference) => Ordering::Greater,
            (TermType::Closure, _) => Ordering::Less,
            (_, TermType::Closure) => Ordering::Greater,
            (TermType::Port, _) => Ordering::Less,
            (_, TermType::Port) => Ordering::Greater,
            (TermType::Pid, _) => Ordering::Less,
            (_, TermType::Pid) => Ordering::Greater,
            (TermType::Tuple, _) => Ordering::Less,
            (_, TermType::Tuple) => Ordering::Greater,
            (TermType::Map, _) => Ordering::Less,
            (_, TermType::Map) => Ordering::Greater,
            (TermType::Nil, _) => Ordering::Less,
            (_, TermType::Nil) => Ordering::Greater,
            (TermType::Cons, _) => Ordering::Less,
            (_, TermType::Cons) => Ordering::Greater,
            (TermType::Binary, _) => Ordering::Less,
            (_, TermType::Binary) => Ordering::Greater,
            (TermType::Match, _) => Ordering::Less,
            (_, TermType::Match) => Ordering::Greater,
            (TermType::Code, _) => Ordering::Less,
            (_, TermType::Code) => Ordering::Greater,
            (TermType::Catch, _) => Ordering::Less,
            (_, TermType::Catch) => Ordering::Greater,
            (TermType::Header, _) => Ordering::Less,
            (_, TermType::Header) => Ordering::Greater,
            (TermType::Hole, _) => Ordering::Less,
            (_, TermType::Hole) => Ordering::Greater,
            _ => unreachable!(),
        }
    }
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
/// Pointer values encoded in a term must always have at least 8-byte alignment on all supported
/// platforms. This should be ensured by specifying the required minimum alignment on all concrete
/// term types we define, but we also add some debug checks to protect against accidentally
/// attempting to encode invalid pointers.
///
/// The set of types given explicit type tags were selected such that the most commonly used types
/// are the cheapest to type check and decode. In general, we believe the most used to be numbers,
/// atoms, lists, and tuples.
#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct OpaqueTerm(u64);
impl fmt::Debug for OpaqueTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:064b}", &self.0)
    }
}
impl fmt::Display for OpaqueTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.r#typeof() {
            TermType::Invalid => f.write_str("INVALID"),
            TermType::None => f.write_str("NONE"),
            TermType::Hole => f.write_str("HOLE"),
            TermType::Code => write!(f, "Code({:p})", unsafe { self.as_ptr() }),
            TermType::Catch => write!(f, "Catch({:p})", unsafe { self.as_ptr() }),
            TermType::Header => write!(f, "{:?}", unsafe { self.as_header() }),
            TermType::Match => write!(f, "{:?}", unsafe {
                &*(self.as_ptr() as *const MatchContext)
            }),
            _ => {
                let t: Term = (*self).into();
                write!(f, "{}", &t)
            }
        }
    }
}
impl crate::cmp::ExactEq for OpaqueTerm {
    fn exact_eq(&self, other: &Self) -> bool {
        // Trivial equality
        if self.eq(other) {
            return true;
        }
        let lhs: Term = (*self).into();
        let rhs: Term = (*other).into();
        lhs.exact_eq(&rhs)
    }
}
impl OpaqueTerm {
    /// Represents the constant value used to signal an invalid term/exception
    pub const NONE: Self = Self(NONE);
    /// Represents the constant value associated with the value of an empty list
    pub const NIL: Self = Self(NIL);
    /// Represents the constant value 0 encoded as an integer
    pub const ZERO: Self = Self(NEG_INFINITY);
    /// Represents the constant value 'false'
    pub const FALSE: Self = Self(FALSE);
    /// Represents the constant value 'true'
    pub const TRUE: Self = Self(TRUE);
    /// The total number of bits that a small integer can use in an OpaqueTerm
    pub const INT_BITSIZE: usize = INT_BITSIZE;

    /// Returns this opaque term as a raw u64
    #[inline(always)]
    pub const fn raw(self) -> u64 {
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
            other => match other & GROUP_MASK {
                INTEGER_TAG => term.write(Term::Int(value.as_integer())),
                _ => match other & SUBGROUP_MASK {
                    SPECIAL_TAG => match other & HEADER_TAG {
                        CATCH_TAG => term.write(Term::Catch(value.as_ptr() as usize >> 3)),
                        CODE_TAG => term.write(Term::Code(value.as_ptr() as usize >> 3)),
                        // Header and Hole values are not valid Terms
                        _tag => return false,
                    },
                    INFINITY => match other & TAG_MASK {
                        ATOM_TAG => term.write(Term::Atom(value.as_atom())),
                        CONS_TAG | CONS_LITERAL_TAG => {
                            let mut cell = Gc::<Cons>::from_raw_parts(value.as_ptr(), ());
                            loop {
                                if cell.is_move_marker() {
                                    cell = cell.forwarded_to();
                                } else {
                                    term.write(Term::Cons(cell));
                                    break;
                                }
                            }
                        }
                        TUPLE_TAG | TUPLE_LITERAL_TAG => {
                            let value = value;
                            let mut header_ptr = value.as_ptr().cast::<OpaqueTerm>();
                            loop {
                                let header = *header_ptr;
                                if header.is_header() {
                                    let ptr = Gc::<Tuple>::from_raw_parts(
                                        header_ptr.cast(),
                                        header.arity(),
                                    );
                                    term.write(Term::Tuple(ptr));
                                    break;
                                } else {
                                    header_ptr = header.as_ptr().cast();
                                }
                            }
                        }
                        RC_TAG => {
                            let ptr = value.as_ptr();
                            // When we decode a reference-counted term, we clone a new strong
                            // reference. This means that
                            // reference-counted terms won't be deallocated while an OpaqueTerm
                            // containing the reference is held on the process heap or stack, until
                            // a GC occurs
                            let header = *ptr.cast::<Header>();
                            match header.tag() {
                                Tag::Port => {
                                    let rc = Arc::<Port>::from_raw(ptr.cast());
                                    term.write(Term::Port(Arc::clone(&rc)));
                                    mem::forget(rc);
                                }
                                Tag::Binary => {
                                    let bin = <BinaryData as Boxable>::from_raw_parts(ptr, header);
                                    let rc = Arc::<BinaryData>::from_raw(bin);
                                    term.write(Term::RcBinary(Arc::clone(&rc)));
                                    mem::forget(rc);
                                }
                                _ => return false,
                            }
                        }
                        LITERAL_TAG => {
                            // Currently, the only possible type which can be flagged as literal
                            // without any other identifying information
                            // is constant BinaryData.
                            let ptr = value.as_ptr();
                            let ptr = unsafe {
                                <BinaryData as Boxable>::from_raw_parts(ptr, *ptr.cast::<Header>())
                            };
                            // Constant binaries are allocated with a leading usize containing the
                            // pointer metadata
                            term.write(Term::ConstantBinary(&*ptr));
                        }
                        0 => {
                            // This is a Gc
                            let mut ptr = value.as_ptr();
                            loop {
                                let header = *ptr.cast::<OpaqueTerm>();
                                if header.is_header() {
                                    let header = Header::from(header);
                                    match header.tag() {
                                        Tag::BigInt => {
                                            let ptr =
                                                <BigInt as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::BigInt(Gc::from_raw(ptr)));
                                        }
                                        Tag::Tuple => {
                                            let ptr =
                                                <Tuple as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::Tuple(Gc::from_raw(ptr)));
                                        }
                                        Tag::Map => {
                                            let ptr = <Map as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::Map(Gc::from_raw(ptr)));
                                        }
                                        Tag::Closure => {
                                            let ptr =
                                                <Closure as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::Closure(Gc::from_raw(ptr)));
                                        }
                                        Tag::Pid => {
                                            let ptr = <Pid as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::Pid(Gc::from_raw(ptr)));
                                        }
                                        Tag::Port => {
                                            // TODO:
                                            // term.write(Term::Port(Gc::from_raw_parts(ptr,
                                            // header.arity())));
                                            todo!()
                                        }
                                        Tag::Reference => {
                                            let ptr =
                                                <Reference as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::Reference(Gc::from_raw(ptr)));
                                        }
                                        Tag::Binary => {
                                            let ptr = <BinaryData as Boxable>::from_raw_parts(
                                                ptr, header,
                                            );
                                            term.write(Term::HeapBinary(Gc::from_raw(ptr)));
                                        }
                                        Tag::Slice => {
                                            let ptr =
                                                <BitSlice as Boxable>::from_raw_parts(ptr, header);
                                            term.write(Term::RefBinary(Gc::from_raw(ptr)));
                                        }
                                        Tag::Match => return false,
                                    }
                                    break;
                                } else {
                                    // This is a forwarded pointer
                                    assert!(header.is_gcbox());
                                    ptr = header.as_ptr();
                                }
                            }
                        }
                        _tag => return false,
                    },
                    _ => {
                        term.write(Term::Float(value.as_float().into()));
                    }
                },
            },
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
            other => match other & GROUP_MASK {
                INTEGER_TAG => TermType::Int,
                _ => match other & SUBGROUP_MASK {
                    // The catch/code types are considered invalid for this function
                    SPECIAL_TAG => match other & HEADER_TAG {
                        0 if other & HOLE_TAG == HOLE_TAG => TermType::Hole,
                        CATCH_TAG => TermType::Catch,
                        CODE_TAG => TermType::Code,
                        HEADER_TAG => TermType::Header,
                        _tag => TermType::Invalid,
                    },
                    INFINITY => {
                        match other & TAG_MASK {
                            ATOM_TAG => TermType::Atom,
                            CONS_TAG | CONS_LITERAL_TAG => TermType::Cons,
                            TUPLE_TAG | TUPLE_LITERAL_TAG => TermType::Tuple,
                            RC_TAG => {
                                let ptr = unsafe { self.as_ptr() };
                                let header = unsafe { *ptr.cast::<Header>() };
                                match header.tag() {
                                    Tag::Port => TermType::Port,
                                    Tag::Binary => TermType::Binary,
                                    _ => TermType::Invalid,
                                }
                            }
                            LITERAL_TAG => {
                                // Currently, the only possible type which can be flagged as literal
                                // without any other identifying
                                // information is constant BinaryData.
                                TermType::Binary
                            }
                            0 => {
                                let mut ptr = unsafe { self.as_ptr() };
                                loop {
                                    let header = unsafe { *ptr.cast::<OpaqueTerm>() };
                                    if header.is_header() {
                                        return match unsafe { header.tag() } {
                                            Tag::BigInt => TermType::Int,
                                            Tag::Tuple => TermType::Tuple,
                                            Tag::Map => TermType::Map,
                                            Tag::Closure => TermType::Closure,
                                            Tag::Pid => TermType::Pid,
                                            Tag::Port => TermType::Port,
                                            Tag::Reference => TermType::Reference,
                                            Tag::Binary | Tag::Slice => TermType::Binary,
                                            Tag::Match => TermType::Match,
                                        };
                                    } else {
                                        // This term is forwarded
                                        ptr = unsafe { header.as_ptr() };
                                    }
                                }
                            }
                            _ => TermType::Invalid,
                        }
                    }
                    _ => TermType::Float,
                },
            },
        }
    }

    #[inline]
    pub const fn code(ip: usize) -> Self {
        let ip = (ip as u64) << 3;
        if ip & SUBGROUP_MASK != 0 {
            panic!("invalid code pointer, value too large");
        }
        Self(ip | CODE_TAG | SPECIAL_TAG)
    }

    #[inline]
    pub const fn catch(ip: usize) -> Self {
        let ip = (ip as u64) << 3;
        if ip & SUBGROUP_MASK != 0 {
            panic!("invalid catch pointer, value too large");
        }
        Self(ip | CATCH_TAG | SPECIAL_TAG)
    }

    #[inline]
    pub const fn header(tag: Tag, arity: usize) -> Self {
        let arity = (arity as u64) << 6;
        if arity & SUBGROUP_MASK != 0 {
            panic!("invalid arity value for header, out of range");
        }
        Self(((tag as u64) << 2) | arity | SPECIAL_TAG | HEADER_TAG)
    }

    /// Returns true if this value is not a valid floating point number (i.e. NaN or infinity)
    #[inline(always)]
    const fn is_nan(self) -> bool {
        self.0 & INFINITY == INFINITY
    }

    /// Returns true if this value is a boxed term header value
    #[inline(always)]
    pub const fn is_header(self) -> bool {
        const SPECIAL_TAG_HDR: u64 = SPECIAL_TAG | HEADER_TAG;
        self.0 & SPECIAL_TAG_MASK == SPECIAL_TAG_HDR
    }

    /// Returns true if this value is a heap hole marker
    pub const fn is_hole(self) -> bool {
        const SPECIAL_TAG_HOLE: u64 = SPECIAL_TAG | 0x08;
        const EXTENDED_SPECIAL_TAG_MASK: u64 = SPECIAL_TAG | TAG_MASK | 0x08;
        self.0 & EXTENDED_SPECIAL_TAG_MASK == SPECIAL_TAG_HOLE
    }

    /// Creates a new hole marker value with the given size
    pub fn hole(size: usize) -> Self {
        let size = (size as u64)
            .checked_shl(4)
            .expect("invalid hole size, value out of range");
        assert_eq!(size & SUBGROUP_MASK, 0);
        Self(size | SPECIAL_TAG | 0x08)
    }

    /// This function checks if the current term is a pointer to a moved term, and
    /// returns the forwarded pointer, or `None` if the term is not a move marker.
    ///
    /// There are three different types of move markers, corresponding to three
    /// of the four unique types of boxed values:
    ///
    /// * Cons cells, which are rewritten such that the cell head is NONE and the
    /// cell tail is a cons pointer to the new storage location.
    /// * Tuples, which have their header term rewritten as a tuple pointer to the
    /// new storage location
    /// * Gc<T>, which have their type id rewritten to that of the ForwardingMarker
    /// type, and contain the forwarding address in the box metadata.
    ///
    /// The final type of boxed term, using Arc<T>, are reference-counted rather than
    /// cloned/moved, so they have no need for move markers.
    pub fn move_marker(self) -> Option<NonNull<()>> {
        if self.is_nonempty_list() {
            let ptr = unsafe { self.as_ptr() };
            // Move marker for cons cells rewrites the cell head as NONE and cell tail as a cons
            // pointer
            let cons = unsafe { &*ptr.cast::<Cons>() };
            if cons.is_move_marker() {
                Some(unsafe { cons.forwarded_to().as_non_null_ptr().cast() })
            } else {
                None
            }
        } else if self.is_tuple() {
            let ptr = unsafe { self.as_ptr() };
            // Move marker for tuples rewrites the header term as a tuple pointer
            let header = unsafe { *ptr.cast::<OpaqueTerm>() };
            if header.is_header() {
                None
            } else {
                assert!(header.is_tuple());
                Some(NonNull::new(unsafe { header.as_ptr() }).unwrap())
            }
        } else if self.is_gcbox() {
            let ptr = unsafe { self.as_ptr() };
            let header = unsafe { *ptr.cast::<OpaqueTerm>() };
            if header.is_header() {
                None
            } else {
                Some(unsafe { NonNull::new_unchecked(header.as_ptr()) })
            }
        } else {
            None
        }
    }

    /// Returns the tag associated with this header term
    ///
    /// # SAFETY
    ///
    /// In debug mode this function will panic if the term is not a header,
    /// but in releases this must be guaranteed by the caller.
    #[inline]
    pub unsafe fn tag(self) -> Tag {
        debug_assert!(self.is_header());
        let tag = (self.0 & 0b111100) as u8;
        unsafe { core::mem::transmute::<u8, Tag>(tag >> 2) }
    }

    /// Returns the arity value associated with this header term
    ///
    /// # SAFETY
    ///
    /// In debug mode this function will panic if the term is not a header,
    /// but in releases this must be guaranteed by the caller.
    #[inline]
    pub const unsafe fn arity(self) -> usize {
        debug_assert!(self.is_header());
        ((self.0 & PTR_MASK) >> 6) as usize
    }

    /// Returns the size of the hole indicated by a hole marker
    pub unsafe fn hole_size(self) -> usize {
        debug_assert!(self.is_hole());
        const HOLE_SIZE_MASK: u64 = PTR_MASK & !0x08;
        ((self.0 & HOLE_SIZE_MASK) >> 4) as usize
    }

    /// Returns true if this term is a non-boxed value
    ///
    /// This returns true for floats, small integers, nil, and atoms
    ///
    /// NOTE: This returns false for None, as None is not a valid term value
    #[inline(always)]
    pub fn is_immediate(self) -> bool {
        !self.is_special() && !self.is_box()
    }

    /// Returns true if this term is a special marker value and not a valid term
    #[inline]
    pub fn is_special(self) -> bool {
        // This is guaranteed to be unique to special marker values
        self.0 & SUBGROUP_MASK == SPECIAL_TAG
    }

    /// Returns true if this term is a code pointer
    #[inline]
    pub fn is_code(self) -> bool {
        const SPECIAL_TAG_CODE: u64 = SPECIAL_TAG | CODE_TAG;
        self.0 & SPECIAL_TAG_MASK == SPECIAL_TAG_CODE
    }

    /// Returns true if this term is a catch pointer
    #[inline]
    pub fn is_catch(self) -> bool {
        const SPECIAL_TAG_CATCH: u64 = SPECIAL_TAG | CATCH_TAG;
        self.0 & SPECIAL_TAG_MASK == SPECIAL_TAG_CATCH
    }

    /// Returns true if this term is a pointer to a match context
    pub fn is_match_context(self) -> bool {
        if !self.is_gcbox() {
            return false;
        }
        let header = unsafe { &*self.as_ptr().cast::<Header>() };
        header.tag() == Tag::Match
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
    pub const fn is_box(self) -> bool {
        match self.0 {
            // This is a singleton immediate
            NIL | TRUE | FALSE => false,
            other => match other & GROUP_MASK {
                INTEGER_TAG => false,
                _ => match other & SUBGROUP_MASK {
                    // This is a special marker value
                    SPECIAL_TAG => false,
                    // Need to check the tag bits to identify immediate vs boxed
                    INFINITY => match other & TAG_MASK {
                        // Gc<T>, Literal, Arc<T>, Cons, Tuple
                        0 | LITERAL_TAG | RC_TAG | CONS_TAG | CONS_LITERAL_TAG | TUPLE_TAG
                        | TUPLE_LITERAL_TAG => true,
                        _ => false,
                    },
                    // This is a float or integer immediate
                    _ => false,
                },
            },
        }
    }

    /// Returns true if this term is a non-null pointer to a Gc<T> term
    #[inline]
    pub fn is_gcbox(self) -> bool {
        self.0 & (SUBGROUP_MASK | TAG_MASK) == INFINITY && self.0 != NIL
    }

    /// Returns true if this term is a non-null pointer to a Arc<T> term
    #[inline]
    pub fn is_rc(self) -> bool {
        self.0 & (SUBGROUP_MASK | TAG_MASK) == (INFINITY | RC_TAG) && self.0 != TRUE
    }

    /// Returns true if this term is a non-null pointer to a literal term
    #[inline]
    pub fn is_literal(self) -> bool {
        self.0 & (SUBGROUP_MASK | TAG_MASK) == (INFINITY | LITERAL_TAG)
    }

    /// Returns true if the underlying bit pattern for this term is all zeroes
    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0 == 0
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
        match self.0 {
            TRUE | FALSE => true,
            other if other & (SUBGROUP_MASK | TAG_MASK) == (INFINITY | ATOM_TAG) => true,
            _ => false,
        }
    }

    /// Returns true only if this term is an immediate integer
    ///
    /// NOTE: This does not return true for big integers
    #[inline(always)]
    pub fn is_integer(self) -> bool {
        self.0 & GROUP_MASK == INTEGER_TAG
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

        match self.0 & (SUBGROUP_MASK | TAG_MASK) {
            IS_CONS | IS_CONS_LITERAL => true,
            _ => false,
        }
    }

    /// Returns true if this term is nil or a cons cell pointer
    #[inline]
    pub fn is_list(self) -> bool {
        self.0 == NIL || self.is_nonempty_list()
    }

    /// This is a specialized type check for the case where we want to distinguish
    /// if a term is a pointer to one of the two special pointees which are not allocated
    /// via Gc, cons cells and tuples.
    #[inline]
    pub fn is_cons_or_tuple(self) -> bool {
        const IS_CONS: u64 = INFINITY | CONS_TAG;
        const IS_CONS_LITERAL: u64 = INFINITY | CONS_LITERAL_TAG;

        const IS_TUPLE: u64 = INFINITY | TUPLE_TAG;
        const IS_TUPLE_LITERAL: u64 = INFINITY | TUPLE_LITERAL_TAG;

        match self.0 & (SUBGROUP_MASK | TAG_MASK) {
            IS_CONS | IS_CONS_LITERAL | IS_TUPLE | IS_TUPLE_LITERAL => true,
            _ => false,
        }
    }

    /// Returns true if this term is a tuple pointer
    #[inline]
    pub fn is_tuple(self) -> bool {
        const IS_TUPLE: u64 = INFINITY | TUPLE_TAG;
        const IS_TUPLE_LITERAL: u64 = INFINITY | TUPLE_LITERAL_TAG;

        match self.0 & (SUBGROUP_MASK | TAG_MASK) {
            IS_TUPLE | IS_TUPLE_LITERAL => true,
            _ => false,
        }
    }

    /// Returns true if this term is a tuple with the given arity
    #[inline]
    pub fn is_tuple_with_arity(self, arity: u32) -> bool {
        match self.tuple_size() {
            Ok(n) => arity == n,
            Err(_) => false,
        }
    }

    /// Returns true if this term is a container type for other terms (list/tuple/map/closure)
    pub fn is_container(self) -> bool {
        match self.r#typeof() {
            TermType::Cons | TermType::Tuple | TermType::Map | TermType::Closure => true,
            _ => false,
        }
    }

    /// A combined tuple type test with fetching the arity, optimized for a specific pattern
    /// produced by the compiler
    pub fn tuple_size(self) -> Result<u32, ()> {
        const IS_TUPLE: u64 = INFINITY | TUPLE_TAG;
        const IS_TUPLE_LITERAL: u64 = INFINITY | TUPLE_LITERAL_TAG;

        match self.0 & (SUBGROUP_MASK | TAG_MASK) {
            IS_TUPLE | IS_TUPLE_LITERAL => {
                let header = unsafe { *self.as_ptr().cast::<Header>() };
                Ok(header.arity() as u32)
            }
            _ => Err(()),
        }
    }

    /// Like `erlang:size/1`, but returns the dynamic size of the given term, or 0 if it is not an
    /// unsized type
    ///
    /// For tuples, this is the number of elements in the tuple.
    /// For closures, it is the number of elements in the closure environment.
    /// FOr binaries/bitstrings, it is the size in bytes.
    pub fn size(self) -> usize {
        use firefly_binary::Bitstring;
        match self.into() {
            Term::Tuple(tup) => tup.len(),
            Term::Map(map) => map.size(),
            Term::Closure(fun) => fun.env_size(),
            Term::HeapBinary(bin) => bin.byte_size(),
            Term::RcBinary(bin) => bin.byte_size(),
            Term::RefBinary(slice) => slice.byte_size(),
            Term::ConstantBinary(bin) => bin.byte_size(),
            _ => 0,
        }
    }

    /// Returns the `BinaryFlags` for this term if it is a binary/bitstring
    pub fn binary_flags(self) -> Option<firefly_binary::BinaryFlags> {
        use firefly_binary::{Binary, BinaryFlags, Bitstring, Encoding};
        match self.into() {
            Term::HeapBinary(bin) => Some(bin.flags()),
            Term::RcBinary(bin) => Some(bin.flags()),
            Term::RefBinary(bin) => {
                let size = bin.byte_size();
                Some(
                    BinaryFlags::new(size, Encoding::Raw)
                        .with_trailing_bits(bin.trailing_bits() as usize),
                )
            }
            Term::ConstantBinary(bin) => Some(bin.flags()),
            _ => None,
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
    pub const unsafe fn as_ptr(self) -> *mut () {
        debug_assert!(self.is_box());

        (self.0 & PTR_MASK) as *mut ()
    }

    /// Returns a `Header` decoded from this opaque term
    #[inline]
    pub unsafe fn as_header(self) -> Header {
        debug_assert!(self.is_header());
        Header::from(self)
    }

    /// Extracts the raw catch pointer from this term
    #[inline]
    pub unsafe fn as_catch(self) -> usize {
        debug_assert!(self.is_catch());
        ((self.0 & PTR_MASK) >> 3).try_into().unwrap()
    }

    /// Extracts the raw code pointer from this term
    #[inline]
    pub fn as_code(self) -> usize {
        debug_assert!(self.is_code());
        ((self.0 & PTR_MASK) >> 3).try_into().unwrap()
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
                assert!(!ptr.is_null());
                let ptr = unsafe { NonNull::new_unchecked(ptr) };
                ptr.into()
            }
        }
    }

    /// Extracts the integer value contained in this term.
    ///
    /// This function is always memory safe, but if improperly used will cause weird results, so it
    /// is important that you guard usages of this function with proper type checks.
    pub unsafe fn as_integer(self) -> i64 {
        debug_assert!(self.is_integer());
        // Extract the raw 51-bit signed integer
        let raw = self.0 & INT_MASK;
        // Sign-extend to 64-bits by multiplying the extra bits by 1 if signed, 0 if unsigned
        let sign = ((raw & SIGN_BIT == SIGN_BIT) as u64) * NEG_INTEGER_TAG;
        (raw | sign) as i64
    }

    /// Convert this term to a floating-point value without any type checks
    ///
    /// This is always safe as all our non-float values are encoded as NaN
    pub fn as_float(self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Returns true if the given i64 value is in the range allowed for immediates
    pub fn is_small_integer(value: i64) -> bool {
        let value = value as u64;
        match value & INTEGER_TAG {
            0 | INTEGER_TAG => true,
            _ => false,
        }
    }

    /// This function can be called when cloning a term that might be reference-counted
    pub fn maybe_increment_refcount(&self) -> bool {
        if self.is_rc() {
            // We don't need to cast to a concrete type, as it does not matter for this operation
            unsafe {
                Arc::<()>::increment_strong_count(self.as_ptr());
            }
            true
        } else {
            false
        }
    }

    /// This function can be called when dropping a term that might be reference-counted
    pub fn maybe_decrement_refcount(&self) -> bool {
        if self.is_rc() {
            unsafe {
                Arc::<()>::decrement_strong_count(self.as_ptr());
            }
            true
        } else {
            false
        }
    }

    /// This function is here to allow the Fn/FnMut/etc. impls to properly re-encode the
    /// callee term when applied from Rust. We currently only allow Closure to be allocated
    /// via Gc, so this is safe as long as that holds true, but we explicitly don't implement
    /// From for this conversion, as it is inherently unsafe.
    ///
    /// NOTE: The encoding here matches that of Gc<Closure>, which should be exactly equivalent
    /// as long as the closure being given here was originally allocated via Gc and not via other
    /// means.
    pub unsafe fn from_gcbox_closure(closure: *const Closure) -> Self {
        let (raw, _) = closure.to_raw_parts();
        let raw = raw as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY)
    }
}
impl PartialOrd for OpaqueTerm {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OpaqueTerm {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        if self.eq(other) {
            return Ordering::Equal;
        }

        // None and the various special types must always be sorted last
        if self.is_none() || self.is_special() {
            return Ordering::Greater;
        }
        if other.is_none() || other.is_special() {
            return Ordering::Less;
        }

        // Before decoding as Term, try to compare based just on the term type
        let a = self.r#typeof();
        let b = other.r#typeof();
        let result = a.cmp(&b);
        if result != Ordering::Equal {
            return result;
        }

        // The term types are equal, so delegate to Term for the more expensive comparison check
        let a: Term = (*self).into();
        let b: Term = (*other).into();
        a.cmp(&b)
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
        let u = i as u64;
        match u & NEG_INTEGER_TAG {
            // Positive integer
            0 => Ok(Self(u | INTEGER_TAG)),
            // Negative integer
            NEG_INTEGER_TAG => Ok(Self(u)),
            // Out of range
            _ => Err(ImmediateOutOfRangeError),
        }
    }
}
impl From<f64> for OpaqueTerm {
    #[inline]
    fn from(f: f64) -> Self {
        assert!(!f.is_infinite());
        Self(f.to_bits())
    }
}
impl From<Float> for OpaqueTerm {
    #[inline]
    fn from(f: Float) -> Self {
        f.inner().into()
    }
}
impl From<Atom> for OpaqueTerm {
    #[inline]
    fn from(a: Atom) -> Self {
        if a == atoms::True {
            Self::TRUE
        } else if a == atoms::False {
            Self::FALSE
        } else {
            Self(unsafe { a.as_ptr() as u64 | FALSE })
        }
    }
}
impl<T: ?Sized> From<Gc<T>> for OpaqueTerm {
    default fn from(boxed: Gc<T>) -> Self {
        let (raw, _) = boxed.to_raw_parts();
        let raw = raw as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY)
    }
}
impl<T: ?Sized> From<Arc<T>> for OpaqueTerm {
    fn from(boxed: Arc<T>) -> Self {
        let raw = Arc::into_raw(boxed).cast::<()>() as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
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
        let (raw, _) = ptr.to_raw_parts();
        let raw = raw.as_ptr() as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | CONS_TAG)
    }
}
impl From<Gc<Cons>> for OpaqueTerm {
    fn from(ptr: Gc<Cons>) -> Self {
        let (raw, _) = ptr.to_raw_parts();
        let raw = raw as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
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
        let (raw, _meta) = ptr.to_raw_parts();
        let raw = raw.as_ptr() as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
            "expected nan bits to be unused in pointers"
        );
        debug_assert!(
            raw & TAG_MASK == 0,
            "expected pointer to have at least 8-byte alignment"
        );
        Self(raw | INFINITY | TUPLE_TAG)
    }
}
impl From<Gc<Tuple>> for OpaqueTerm {
    fn from(ptr: Gc<Tuple>) -> Self {
        let (raw, _meta) = ptr.to_raw_parts();
        let raw = raw as u64;
        debug_assert!(
            raw & SUBGROUP_MASK == 0,
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
            raw & SUBGROUP_MASK == 0,
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
            Term::Catch(c) => Self::catch(c),
            Term::Code(c) => Self::code(c),
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
            Term::RcBinary(rc) => rc.into(),
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
impl TryInto<char> for OpaqueTerm {
    type Error = ();

    fn try_into(self) -> Result<char, Self::Error> {
        match self.into() {
            Term::Int(i) => match i.try_into() {
                Ok(cp) => char::from_u32(cp).ok_or(()),
                Err(_) => Err(()),
            },
            _ => Err(()),
        }
    }
}
impl PartialEq<Atom> for OpaqueTerm {
    fn eq(&self, other: &Atom) -> bool {
        let atom: Self = (*other).into();
        atom.eq(self)
    }
}
impl firefly_system::sync::Atom for OpaqueTerm {
    type Repr = u64;

    #[inline]
    fn pack(self) -> Self::Repr {
        self.0
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        Self(raw)
    }
}

#[cfg(test)]
mod tests {
    use core::assert_matches::assert_matches;

    use alloc::alloc::Layout;
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::ffi::c_void;
    use core::mem::MaybeUninit;
    use core::ptr;

    use firefly_alloc::heap::FixedSizeHeap;
    use firefly_arena::DroplessArena;
    use firefly_binary::{BinaryFlags, Bitstring, Encoding, Selection};

    use crate::drivers::{self, Driver, DriverError, DriverFlags, LoadableDriver};
    use crate::function::ErlangResult;
    use crate::gc::Gc;
    use crate::process::ProcessId;
    use crate::scheduler::SchedulerId;
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
            let (layout, value_offset) = Layout::new::<Header>()
                .extend(Layout::array::<u8>(size).unwrap())
                .unwrap();
            let layout = layout.pad_to_align();
            let ptr = self.arena.alloc_raw(layout);

            // Write flags
            let header_ptr = ptr.cast::<Header>();
            let flags = BinaryFlags::new(size, Encoding::detect(bytes));
            header_ptr.write(Header::new(Tag::Binary, flags.into_raw()));

            // Write data
            let bytes_ptr: *mut u8 = (header_ptr as *mut u8).add(value_offset);
            ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_ptr, size);

            // Reify as static reference
            let data_ptr: *const BinaryData = ptr::from_raw_parts(ptr.cast(), size);
            &*data_ptr
        }
    }

    #[derive(Copy, Clone)]
    struct TestDriver;

    struct TestDriverState {
        #[allow(unused)]
        port: Arc<Port>,
    }

    impl LoadableDriver for TestDriver {
        fn init(&self) -> Result<(), DriverError> {
            Ok(())
        }
        fn name(&self) -> &str {
            "test_driver"
        }
        fn version(&self) -> (u32, u32) {
            (1, 0)
        }
        fn flags(&self) -> DriverFlags {
            DriverFlags::default()
        }
        fn start(
            &self,
            port: Arc<MaybeUninit<Port>>,
            _command: &str,
        ) -> Result<Box<dyn Driver>, DriverError> {
            Ok(Box::new(TestDriverState {
                port: unsafe { port.assume_init() },
            }))
        }
    }

    impl Driver for TestDriverState {
        fn stop(&self) {
            unreachable!()
        }
        fn output(&self, _buffer: &[u8]) {
            unreachable!()
        }
        fn ready_input(&self, _event: *mut ()) {
            unreachable!()
        }
        fn ready_output(&self, _event: *mut ()) {
            unreachable!()
        }
        fn control(&self, _command: u32, _buf: &[u8], _rbuf: *mut *mut u8, _rlen: usize) -> usize {
            unreachable!()
        }
        fn timeout(&self) {
            unreachable!()
        }
        #[cfg(feature = "std")]
        fn outputv<'a>(&self, _data: std::io::IoSlice<'a>) {
            unreachable!()
        }
        #[cfg(not(feature = "std"))]
        fn outputv<'a>(&self, _data: &[&'a [u8]]) {
            unreachable!()
        }
        fn ready_async(&self, _async_data: *mut c_void) {
            unreachable!()
        }
        fn flush(&self) {
            unreachable!()
        }
        fn call(
            &self,
            _command: u32,
            _buf: &[u8],
            _rbuf: *mut *mut u8,
            _rlen: usize,
            _flags: *mut u32,
        ) -> Result<usize, DriverError> {
            unreachable!()
        }
        fn process_exit(&self, _monitor: drivers::DriverMonitor) {
            unreachable!()
        }
        fn stop_select(&self, _event: drivers::DriverEvent, _reserved: *mut ()) {
            unreachable!()
        }
    }

    #[test]
    fn opaque_term_decode() {
        let heap = FixedSizeHeap::<1024>::default();

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
        let cons = Cons::new_in(
            Cons {
                head: OpaqueTerm::NIL,
                tail: OpaqueTerm::NIL,
            },
            &heap,
        )
        .unwrap();
        assert!(unsafe { OpaqueTerm::decode(cons.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Cons(cons));

        // Tuple
        let mut term = MaybeUninit::zeroed();
        let tuple = Tuple::from_slice(&[atoms::Ok.into(), OpaqueTerm::NIL], &heap).unwrap();
        assert!(unsafe { OpaqueTerm::decode(tuple.into(), term.as_mut_ptr()) });
        assert_eq!(unsafe { term.assume_init() }, Term::Tuple(tuple));

        // Map
        let mut term = MaybeUninit::zeroed();
        let mut map = Map::with_capacity_in(1, &heap).unwrap();
        map.put_mut(Term::Int(1), Term::Bool(true));
        assert!(unsafe { OpaqueTerm::decode(map.into(), term.as_mut_ptr()) });
        let map = unsafe { term.assume_init() };
        assert_matches!(map, Term::Map(_));
        let Term::Map(map) = map else { unreachable!(); };
        let value = map.get(Term::Int(1)).map(|t| t.into());
        assert_eq!(value, Some(Term::Bool(true)));

        // Closure
        let mut term = MaybeUninit::zeroed();
        let fun = erlang_error_1 as *const ();
        let closure = Closure::new_in(atoms::Erlang, atoms::Error, 1, fun, &[], &heap).unwrap();
        assert!(unsafe { OpaqueTerm::decode(closure.into(), term.as_mut_ptr()) });
        let closure = unsafe { term.assume_init() };
        assert_matches!(closure, Term::Closure(_));
        let Term::Closure(closure) = closure else { unreachable!() };
        assert_eq!(closure.callee, fun);

        // Pid
        let mut term = MaybeUninit::zeroed();
        let proc_id = ProcessId::new(1, 1).unwrap();
        let pid = Pid::new_local(proc_id);
        let pid2 = Gc::new_in(pid.clone(), &heap).unwrap();
        assert!(unsafe { OpaqueTerm::decode(pid2.into(), term.as_mut_ptr()) });
        let pid2 = unsafe { term.assume_init() };
        assert_matches!(pid2, Term::Pid(_));
        assert_eq!(pid2, Term::Pid(Gc::new(pid.clone())));

        // Port
        let mut term = MaybeUninit::zeroed();
        let driver = TestDriver;
        driver.init().unwrap();
        let id = PortId::from_raw(1);
        let port = Port::new_with_id(id, pid, "test_driver", &driver).unwrap();
        let port2 = port.clone();
        assert!(unsafe { OpaqueTerm::decode(port2.into(), term.as_mut_ptr()) });
        let port2 = unsafe { term.assume_init() };
        assert_matches!(port2, Term::Port(_));
        assert_eq!(port2, Term::Port(port));

        // Reference
        let mut term = MaybeUninit::zeroed();
        let scheduler_id = unsafe { SchedulerId::from_raw(1) };
        let reference = Reference::new(unsafe { ReferenceId::new(scheduler_id, 1) });
        let reference2 = Gc::new_in(reference.clone(), &heap).unwrap();
        assert!(unsafe { OpaqueTerm::decode(reference2.into(), term.as_mut_ptr()) });
        let reference2 = unsafe { term.assume_init() };
        assert_matches!(reference2, Term::Reference(_));
        assert_eq!(reference2, Term::Reference(Gc::new(reference)));

        // Binary
        let rc = BinaryData::from_str("testing 1 2 3");
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(rc.clone().into(), term.as_mut_ptr()) });
        let rc_term = unsafe { term.assume_init() };
        assert_matches!(rc_term, Term::RcBinary(_));
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(rc.into(), term.as_mut_ptr()) });
        let rc = unsafe { term.assume_init() };
        assert_matches!(rc, Term::RcBinary(_));

        let s = "testing 1 2 3";
        let mut bin = BinaryData::with_capacity_small(s.as_bytes().len(), &heap).unwrap();
        bin.copy_from_slice(s.as_bytes());
        let mut term = MaybeUninit::zeroed();
        assert!(unsafe { OpaqueTerm::decode(bin.into(), term.as_mut_ptr()) });
        let bin = unsafe { term.assume_init() };
        assert_matches!(bin, Term::HeapBinary(_));
        let Term::HeapBinary(bin) = bin else { unreachable!(); };
        assert_eq!(bin.as_str(), Some("testing 1 2 3"));

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
        assert_eq!(selection.as_str(), Some("testing 1 2 3"));
        let bin = Gc::new_in(BitSlice::from_selection(bin.into(), selection), &heap).unwrap();
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

        let heap = FixedSizeHeap::<1024>::default();

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
        let cons = Cons::new_in(
            Cons {
                head: OpaqueTerm::NIL,
                tail: OpaqueTerm::NIL,
            },
            &heap,
        )
        .unwrap();
        let term: OpaqueTerm = cons.into();
        assert_eq!(term.r#typeof(), TermType::Cons);

        // Tuple
        let tuple = Tuple::from_slice(&[atoms::Ok.into(), OpaqueTerm::NIL], &heap).unwrap();
        let term: OpaqueTerm = tuple.into();
        assert_eq!(term.r#typeof(), TermType::Tuple);

        // Map
        let map = Map::new_in(&heap).unwrap();
        let term: OpaqueTerm = map.into();
        assert_eq!(term.r#typeof(), TermType::Map);

        // Closure
        let closure = Closure::new_in(
            atoms::Erlang,
            atoms::Error,
            1,
            erlang_error_1 as *const (),
            &[],
            &heap,
        )
        .unwrap();
        let term: OpaqueTerm = closure.into();
        assert_eq!(term.r#typeof(), TermType::Closure);

        // Pid
        let proc_id = ProcessId::new(1, 1).unwrap();
        let pid = Gc::new_in(Pid::new_local(proc_id), &heap).unwrap();
        let term: OpaqueTerm = pid.clone().into();
        assert_eq!(term.r#typeof(), TermType::Pid);

        // Port
        let driver = TestDriver;
        driver.init().unwrap();
        let id = PortId::from_raw(1);
        let port = Port::new_with_id(id, (*pid).clone(), "test_driver", &driver).unwrap();
        let term: OpaqueTerm = port.into();
        assert_eq!(term.r#typeof(), TermType::Port);

        // Reference
        let scheduler_id = unsafe { SchedulerId::from_raw(1) };
        let reference = Gc::new_in(
            Reference::new(unsafe { ReferenceId::new(scheduler_id, 1) }),
            &heap,
        )
        .unwrap();
        let term: OpaqueTerm = reference.into();
        assert_eq!(term.r#typeof(), TermType::Reference);

        // Binary
        let rc = BinaryData::from_str("testing 1 2 3");
        let rc_term: OpaqueTerm = rc.into();
        assert_eq!(rc_term.r#typeof(), TermType::Binary);

        let s = "testing 1 2 3";
        let gcbox = BinaryData::with_capacity_small(s.as_bytes().len(), &heap).unwrap();
        let box_term: OpaqueTerm = gcbox.into();
        assert_eq!(box_term.r#typeof(), TermType::Binary);

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
        let bin = Gc::new_in(BitSlice::from_selection(term, selection), &heap).unwrap();
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
        assert!(!OpaqueTerm::NONE.is_tuple());
        assert_eq!(OpaqueTerm::NONE.tuple_size(), Err(()));
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
            assert!(!float.is_tuple());
            assert_eq!(float.tuple_size(), Err(()));
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
            assert!(!int.is_tuple());
            assert_eq!(int.tuple_size(), Err(()));
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
            assert!(!atom.is_tuple());
            assert_eq!(atom.tuple_size(), Err(()));
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
        assert!(!term.is_tuple());
        assert_eq!(term.tuple_size(), Err(()));
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
        assert!(!OpaqueTerm::NIL.is_tuple());
        assert_eq!(OpaqueTerm::NIL.tuple_size(), Err(()));
    }

    #[test]
    fn opaque_term_cons() {
        let heap = FixedSizeHeap::<128>::default();
        // A list containing a single empty list, e.g. `[[]]`
        let list = Cons::new_in(
            Cons {
                head: OpaqueTerm::NIL,
                tail: OpaqueTerm::NIL,
            },
            &heap,
        )
        .unwrap();
        let cons: OpaqueTerm = list.into();

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
        assert!(!cons.is_tuple());
        assert_eq!(cons.tuple_size(), Err(()));
    }

    #[test]
    fn opaque_term_tuple() {
        let heap = FixedSizeHeap::<128>::default();

        // A list containing a single empty list, e.g. `[[]]`
        let tuple = Tuple::from_slice(
            &[atoms::True.into(), atoms::False.into(), OpaqueTerm::NIL],
            &heap,
        )
        .unwrap();
        assert_eq!(tuple.len(), 3);
        let opaque: OpaqueTerm = tuple.into();

        let term: Term = opaque.into();
        assert_eq!(Term::Tuple(tuple), term);
        assert!(opaque.is_nan());
        assert!(!opaque.is_nil());
        assert!(!opaque.is_immediate());
        assert!(opaque.is_box());
        assert!(!opaque.is_gcbox());
        assert!(!opaque.is_rc());
        assert!(!opaque.is_literal());
        assert!(!opaque.is_atom());
        assert!(!opaque.is_integer());
        assert!(!opaque.is_float());
        assert!(!opaque.is_number());
        assert!(!opaque.is_nonempty_list());
        assert!(!opaque.is_list());
        assert_eq!(opaque.tuple_size(), Ok(3));
        assert!(opaque.is_tuple());
        assert!(opaque.is_tuple_with_arity(3));
        assert!(!opaque.is_tuple_with_arity(2));
        assert!(!opaque.is_tuple_with_arity(4));
    }

    #[test]
    fn opaque_term_gcbox() {
        let heap = FixedSizeHeap::<128>::default();

        let mut boxed = Map::with_capacity_in(1, &heap).unwrap();
        assert_eq!(boxed.size(), 0);
        assert_eq!(boxed.capacity(), 1);
        boxed.put_mut(Term::Int(1), Term::Atom(atoms::True));
        // Save the raw pointer
        let (ptr, metadata) = boxed.to_raw_parts();
        let boxed = unsafe { Gc::<Map>::from_raw_parts(ptr, metadata) };
        let map: OpaqueTerm = boxed.into();

        assert_eq!(ptr as *mut (), unsafe { map.as_ptr() });
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
        assert!(!map.is_tuple());
        assert_eq!(map.tuple_size(), Err(()));
    }

    #[test]
    fn opaque_term_rcbox() {
        let boxed = BinaryData::from_str("testing 1 2 3");
        // Save the raw pointers
        let rc_ptr = Arc::into_raw(boxed);
        let rc = unsafe { Arc::from_raw(rc_ptr) };
        let rc_bin: OpaqueTerm = rc.into();

        // The pointers should all be to the same object
        assert_eq!(rc_ptr as *mut (), unsafe { rc_bin.as_ptr() });

        for bin in &[rc_bin] {
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
            assert!(!bin.is_tuple());
            assert_eq!(bin.tuple_size(), Err(()));
        }

        let _ = unsafe { Arc::from_raw(rc_ptr) };
    }

    // Used for closure construction
    fn erlang_error_1(a: OpaqueTerm) -> ErlangResult {
        ErlangResult::Ok(a)
    }
}
