#![allow(unused)]
///! This module contains constants for 64-bit architectures used by the term
///! implementation.
///!
///! Currently, both 32-bit and 64-bit architectures rely on a minimum of 8 byte
///! alignment for all terms, as this provides 3 bits for a primary tag in the low
///! end of the pointer/integer value. This is the natural alignment on 64-bit, and
///! while naturally aligned on 32-bit as a result, it does use more address space.
///!
///! Previously, we made use of the high bits on 64-bit, as the current state of the
///! x86_64 platform only uses 48 bits of the 64 available. However, other 64-bit platforms,
///! such as AArch64 and SPARC, have no such restriction and could result in erroneous
///! behavior when compiled for those platforms. Intel is also planning extensions to its
///! processors to use up to 54 bits for addresses, which would cause issues as well.
use core::mem;
use core::fmt;
use core::cmp;

use liblumen_core::sys::sysconf::MIN_ALIGN;
const_assert!(MIN_ALIGN >= 8);

use crate::erts::to_word_size;
use crate::erts::term::prelude::*;

use super::{Tag, Repr};

pub type Word = u64;

const NUM_BITS: u64 = 64;

const TAG_BITS: u64 = 4;
const TAG_SHIFT: u64 = 47;
const TAG_MASK: u64 = 0xFu64 << TAG_SHIFT;

const SUBTAG_SHIFT: u64 = TAG_SHIFT - 4;
const SUBTAG_MASK: u64 = 0xFCu64 << SUBTAG_SHIFT;

const VALUE_MASK: u64 = !(i64::max_value() >> (NUM_BITS - TAG_SHIFT)) as u64;
const VALUE_SHIFT: u64 = 3;

// The highest allowed address in the pointer range
const MAX_ADDR: u64 = (1 << TAG_SHIFT) - 1;
// The beginning of the double range
const MIN_DOUBLE: u64 = !(i64::min_value() >> 12) as u64;
// The mask for the bits containing an immediate value
const IMMEDIATE_MASK: u64 = MAX_ADDR;
// The mask for the bits containing an unshifted small integer
// Small integer values are shifted left VALUE_SHIFT bits when
// encoded, and must be shifted back during decoding, then the
// value must be reinterpreted as a two's complement encoded integer
const SMALL_VALUE: u64 = MAX_ADDR >> VALUE_SHIFT;

// This is the highest assignable aligned address on this architecture
pub const MAX_ALIGNED_ADDR: u64 = MAX_ADDR & !(MIN_ALIGN as u64 - 1);

// The valid range of integer values that can fit in a term with primary tag
pub const MAX_IMMEDIATE_VALUE: u64 = MAX_ADDR;
pub const MAX_ATOM_ID: u64 = MAX_ADDR;

// The valid range of fixed-width integers
pub const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> (NUM_BITS - TAG_SHIFT + VALUE_SHIFT);
pub const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> (NUM_BITS - TAG_SHIFT + VALUE_SHIFT);

// Primary classification:
//
// Float:    Value >= MIN_DOUBLE
// Pointer:  Value <= MAX_ADDR
// None:     Value == 0
// Literal:  Any pointer value where bit 0 is set to 1
// UNUSED:   Value <= MIN_DOUBLE, Tag = 0b0000
// Fixnum:   Value <= MIN_DOUBLE, Tag = 0b0001
// Nil:      Value == 0 + tag,    Tag = 0b0010
// List:     Value <= MIN_DOUBLE, Tag = 0b0011
// Atom:     Value <= MIN_DOUBLE, Tag = 0b0100
// Pid:      Value <= MIN_DOUBLE, Tag = 0b0101
// Port:     Value <= MIN_DOUBLE, Tag = 0b0110
// Tuple:    Value <= MIN_DOUBLE, Tag = 0b0111
// Big+:     Value <= MIN_DOUBLE, Tag = 0b1000
// Map:      Value <= MIN_DOUBLE, Tag = 0b1001
// Ref:      Value <= MIN_DOUBLE, Tag = 0b1010
// Closure:  Value <= MIN_DOUBLE, Tag = 0b1011
// Resource: Value <= MIN_DOUBLE, Tag = 0b1100
// Binary:   Value <= MIN_DOUBLE, Tag = 0b1101
//   Procbin:    Sub-tag = 0b1101_00
//   Heapbin:    Sub-tag = 0b1101_01
//   Subbin:     Sub-tag = 0b1101_10
//   Match Ctx:  Sub-tag = 0b1101_11
// External: Value <= MIN_DOUBLE, Tag = 0b1110
//   Pid:  Sub-tag = 0b1110_00
//   Port: Sub-tag = 0b1110_01
//   Ref:  Sub-tag = 0b1110_10
//
// A tag of 0b1111 is MIN_DOUBLE, so is not free

// Used to mark pointers to literals
const FLAG_LITERAL: u64 = 1;

// We don't tag this type, they are implicit
const FLAG_FLOAT: u64 = u64::max_value();

// Tags
const FLAG_BOXED: u64 = 0;
const FLAG_NONE: u64 = 0;
const FLAG_SMALL_INTEGER: u64 = 1 << TAG_SHIFT;
const FLAG_NIL: u64 = 2 << TAG_SHIFT;
const FLAG_LIST: u64 = 3 << TAG_SHIFT;
const FLAG_ATOM: u64 = 4 << TAG_SHIFT;
const FLAG_PID: u64 = 5 << TAG_SHIFT;
const FLAG_PORT: u64 = 6 << TAG_SHIFT;
// Non-immediate tags
const FLAG_TUPLE: u64 = 7 << TAG_SHIFT;
const FLAG_BIG_INTEGER: u64 = 8 << TAG_SHIFT;
const FLAG_MAP: u64 = 9 << TAG_SHIFT;
const FLAG_REFERENCE: u64 = 10 << TAG_SHIFT;
const FLAG_CLOSURE: u64 = 11 << TAG_SHIFT;
const FLAG_RESOURCE_REFERENCE: u64 = 12 << TAG_SHIFT;

const FLAG_BINARY: u64 = 13 << TAG_SHIFT;
const FLAG_PROCBIN: u64 = 0 | FLAG_BINARY;
const FLAG_HEAPBIN: u64 = (1 << SUBTAG_SHIFT) | FLAG_BINARY;
const FLAG_SUBBINARY: u64 = (2 << SUBTAG_SHIFT) | FLAG_BINARY;
const FLAG_MATCH_CTX: u64 = (3 << SUBTAG_SHIFT) | FLAG_BINARY;

const FLAG_EXTERNAL: u64 = 14 << TAG_SHIFT;
const FLAG_EXTERN_PID: u64 = 0 | FLAG_EXTERNAL;
const FLAG_EXTERN_PORT: u64 = (1 << SUBTAG_SHIFT) | FLAG_EXTERNAL;
const FLAG_EXTERN_REF: u64 = (2 << SUBTAG_SHIFT) | FLAG_EXTERNAL;

const NONE: u64 = 0;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct RawTerm(u64);
impl RawTerm {
    pub const NONE: Self = Self(NONE);
    pub const NIL: Self = Self(FLAG_NIL);

    pub const HEADER_TUPLE: u64 = FLAG_TUPLE;
    pub const HEADER_BIG_INTEGER: u64 = FLAG_BIG_INTEGER;
    pub const HEADER_REFERENCE: u64 = FLAG_REFERENCE;
    pub const HEADER_CLOSURE: u64 = FLAG_CLOSURE;
    pub const HEADER_RESOURCE_REFERENCE: u64 = FLAG_RESOURCE_REFERENCE;
    pub const HEADER_PROCBIN: u64 = FLAG_PROCBIN;
    pub const HEADER_BINARY_LITERAL: u64 = FLAG_PROCBIN;
    pub const HEADER_HEAPBIN: u64 = FLAG_HEAPBIN;
    pub const HEADER_SUBBINARY: u64 = FLAG_SUBBINARY;
    pub const HEADER_MATCH_CTX: u64 = FLAG_MATCH_CTX;
    pub const HEADER_EXTERN_PID: u64 = FLAG_EXTERN_PID;
    pub const HEADER_EXTERN_PORT: u64 = FLAG_EXTERN_PORT;
    pub const HEADER_EXTERN_REF: u64 = FLAG_EXTERN_REF;
    pub const HEADER_MAP: u64 = FLAG_MAP;

    #[inline]
    fn encode_smallint(value: i64) -> Self {
        Self(((value << VALUE_SHIFT) as u64 & IMMEDIATE_MASK) | FLAG_SMALL_INTEGER)
    }

    #[inline]
    fn decode_float(self) -> Float {
        debug_assert!(self.0 >= MIN_DOUBLE);
        Float::new(f64::from_bits(self.0 - MIN_DOUBLE))
    }
}
impl Repr for RawTerm {
    type Word = u64;

    #[inline]
    fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn type_of(self) -> Tag<u64> {
        let term = self.0;
        if term >= MIN_DOUBLE {
            return Tag::Float;
        } else if term == 0 {
            return Tag::None;
        } else if term <= MAX_ADDR {
            return Tag::Box;
        } else {
            // There are 16 available tag combinations, and we use them all for valid values
            match term & TAG_MASK {
                // Binary types have a subtag of 2 bits, all 4 combinations are used
                FLAG_BINARY => match term & SUBTAG_MASK {
                    FLAG_PROCBIN => Tag::ProcBin,
                    FLAG_HEAPBIN => Tag::HeapBinary,
                    FLAG_SUBBINARY => Tag::SubBinary,
                    FLAG_MATCH_CTX => Tag::MatchContext,
                    tag => Tag::Unknown(tag),
                },
                // External types have a subtag of 2 bits, but only 3 combinations are used
                FLAG_EXTERNAL => match term & SUBTAG_MASK {
                    FLAG_EXTERN_PID => Tag::ExternalPid,
                    FLAG_EXTERN_PORT => Tag::ExternalPort,
                    FLAG_EXTERN_REF => Tag::ExternalReference,
                    tag => Tag::Unknown(tag),
                },
                FLAG_TUPLE => Tag::Tuple,
                FLAG_CLOSURE => Tag::Closure,
                FLAG_HEAPBIN => Tag::HeapBinary,
                FLAG_BIG_INTEGER => Tag::BigInteger,
                FLAG_REFERENCE => Tag::Reference,
                FLAG_FLOAT => Tag::Float,
                FLAG_RESOURCE_REFERENCE => Tag::ResourceReference,
                FLAG_PROCBIN => Tag::ProcBin,
                FLAG_SUBBINARY => Tag::SubBinary,
                FLAG_MATCH_CTX => Tag::MatchContext,
                FLAG_EXTERN_PID => Tag::ExternalPid,
                FLAG_EXTERN_PORT => Tag::ExternalPort,
                FLAG_EXTERN_REF => Tag::ExternalReference,
                FLAG_MAP => Tag::Map,
                FLAG_NONE if term == NONE => Tag::None,
                tag => Tag::Unknown(tag)
            }
        }
    }

    #[inline]
    fn encode_immediate(value: u64, tag: u64) -> Self {
        debug_assert!(tag <= TAG_MASK, "invalid primary tag");
        Self((value << VALUE_SHIFT) | tag)
    }

    #[inline]
    fn encode_header(value: u64, tag: u64) -> Self {
        debug_assert!(tag <= SUBTAG_MASK, "invalid header tag");
        Self((value << VALUE_SHIFT) | tag)
    }

    #[inline]
    fn encode_list(value: *const Cons) -> Self {
        Self(value as u64 | FLAG_LIST)
    }

    #[inline]
    fn encode_box<T>(value: *const T) -> Self where T: ?Sized {
        Self(value as *const() as u64 | FLAG_BOXED)
    }

    #[inline]
    fn encode_literal<T>(value: *const T) -> Self where T: ?Sized {
        Self(value as *const() as u64 | FLAG_LITERAL | FLAG_BOXED)
    }

    #[inline]
    unsafe fn decode_list(self) -> Boxed<Cons> {
        debug_assert_eq!(self.0 & TAG_MASK, FLAG_LIST);
        let ptr = (self.0 & !TAG_MASK) as *const Cons as *mut Cons;
        unsafe { Boxed::new_unchecked(ptr) }
    }

    #[inline]
    unsafe fn decode_smallint(self) -> SmallInteger {
        const SMALL_INTEGER_SIGNED: u64 = 1u64 << TAG_SHIFT;

        let value = self.0;
        let tag = value & TAG_MASK;
        let i = if value & SMALL_INTEGER_SIGNED == SMALL_INTEGER_SIGNED {
            !SMALL_VALUE | (value >> VALUE_SHIFT)
        } else {
            SMALL_VALUE & (value >> VALUE_SHIFT)
        } as i64;
        unsafe { SmallInteger::new_unchecked(i as isize) }
    }

    #[inline]
    unsafe fn decode_immediate(self) -> u64 {
        (self.0 & IMMEDIATE_MASK) >> 3
    }

    #[inline]
    unsafe fn decode_atom(self) -> Atom {
        Atom::from_id(self.decode_immediate() as usize)
    }

    #[inline]
    unsafe fn decode_pid(self) -> Pid {
        Pid::from_raw(self.decode_immediate() as usize)
    }

    #[inline]
    unsafe fn decode_port(self) -> Port {
        Port::from_raw(self.decode_immediate() as usize)
    }

    #[inline]
    unsafe fn decode_header_value(&self) -> u64 {
        debug_assert!(self.0 < MIN_DOUBLE, "invalid use of immediate float value as header");
        debug_assert!(self.0 > MAX_ADDR, "invalid use of boxed value as header");
        self.decode_immediate()
    }
}

unsafe impl Send for RawTerm {}

impl Encode<RawTerm> for u8 {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        Ok(RawTerm::encode_immediate((*self) as u64, FLAG_SMALL_INTEGER))
    }
}

impl Encode<RawTerm> for SmallInteger {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        let i: i64 = (*self).into();
        Ok(RawTerm::encode_immediate(i as u64, FLAG_SMALL_INTEGER))
    }
}

impl Encode<RawTerm> for Float {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        Ok(RawTerm(self.value().to_bits() + MIN_DOUBLE))
    }
}

impl Encode<RawTerm> for bool {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        let atom = Atom::try_from_str(&self.to_string()).unwrap();
        Ok(RawTerm::encode_immediate(atom.id() as u64, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Atom {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        Ok(RawTerm::encode_immediate(self.id() as u64, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Pid {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        Ok(RawTerm::encode_immediate(self.as_usize() as u64, FLAG_PID))
    }
}

impl Encode<RawTerm> for Port {
    fn encode(&self) -> Result<RawTerm, TermEncodingError> {
        let value = self.as_usize();
        Ok(RawTerm::encode_immediate(self.as_usize() as u64, FLAG_PORT))
    }
}

impl From<*mut RawTerm> for RawTerm {
    fn from(ptr: *mut RawTerm) -> Self {
        RawTerm::encode_box(ptr)
    }
}

impl_list!(RawTerm);
impl_boxable!(BigInteger, RawTerm);
impl_boxable!(Reference, RawTerm);
impl_boxable!(ExternalPid, RawTerm);
impl_boxable!(ExternalPort, RawTerm);
impl_boxable!(ExternalReference, RawTerm);
impl_boxable!(ResourceReference, RawTerm);
impl_boxable!(Tuple, RawTerm);
impl_boxable!(Map, RawTerm);
impl_boxable!(Closure, RawTerm);
impl_boxable!(ProcBin, RawTerm);
impl_boxable!(HeapBin, RawTerm);
impl_boxable!(SubBinary, RawTerm);
impl_boxable!(MatchContext, RawTerm);
impl_literal!(BinaryLiteral, RawTerm);


impl Cast<*mut RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *mut RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm as *mut RawTerm
    }
}

impl<T> Cast<*mut T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *mut T {
        assert!(self.is_boxed() || self.is_literal());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm as *mut T
    }
}

impl Cast<*mut Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *mut Cons {
        assert!(self.is_list());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm as *mut Cons
    }
}

impl Cast<*const RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *const RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm
    }
}

impl<T> Cast<*const T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *const T {
        assert!(self.is_boxed() || self.is_literal());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const T
    }
}

impl Cast<*const Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *const Cons {
        assert!(self.is_list());
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const Cons
    }
}

impl Encoded for RawTerm {
    #[inline]
    fn decode(&self) -> Result<TypedTerm, TermDecodingError> {
        let tag = self.type_of();
        match tag {
            Tag::Nil => Ok(TypedTerm::Nil),
            Tag::List => Ok(TypedTerm::List(unsafe { self.decode_list() })),
            Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe { self.decode_smallint() })),
            Tag::Float => Ok(TypedTerm::Float(self.decode_float())),
            Tag::Atom => Ok(TypedTerm::Atom(unsafe { self.decode_atom() })),
            Tag::Pid => Ok(TypedTerm::Pid(unsafe { self.decode_pid() })),
            Tag::Port => Ok(TypedTerm::Port(unsafe { self.decode_port() })),
            Tag::Box => {
                let ptr = (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm;
                let unboxed = unsafe { *ptr };
                match unboxed.type_of() {
                    Tag::Nil => Ok(TypedTerm::Nil),
                    Tag::List => Ok(TypedTerm::List(unsafe { unboxed.decode_list() })),
                    Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe { unboxed.decode_smallint() })),
                    Tag::Float => Ok(TypedTerm::Float(unboxed.decode_float())),
                    Tag::Atom => Ok(TypedTerm::Atom(unsafe { unboxed.decode_atom() })),
                    Tag::Pid => Ok(TypedTerm::Pid(unsafe { unboxed.decode_pid() })),
                    Tag::Port => Ok(TypedTerm::Port(unsafe { unboxed.decode_port() })),
                    Tag::Box => Err(TermDecodingError::MoveMarker),
                    Tag::Unknown(_) => Err(TermDecodingError::InvalidTag),
                    Tag::None => Err(TermDecodingError::NoneValue),
                    header => {
                        unboxed.decode_header(header, Some(self.is_literal()))
                    }
                }
            }
            header => self.decode_header(header, None),
        }
    }

    #[inline]
    fn is_none(self) -> bool {
        self.0 == NONE
    }

    #[inline]
    fn is_nil(self) -> bool {
        self.0 == FLAG_NIL
    }

    #[inline]
    fn is_literal(self) -> bool {
        !self.is_float() && (self.0 & FLAG_LITERAL == FLAG_LITERAL)
    }

    #[inline]
    fn is_list(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_LIST)
    }

    #[inline]
    fn is_atom(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_ATOM)
    }

    #[inline]
    fn is_smallint(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_SMALL_INTEGER)
    }

    #[inline]
    fn is_bigint(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_BIG_INTEGER)
    }

    #[inline]
    fn is_integer(&self) -> bool {
        if self.0 >= MIN_DOUBLE {
            return false;
        }
        if self.0 & TAG_MASK == FLAG_SMALL_INTEGER {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::BigInteger(_)) => true,
            _ => false
        }
    }

    #[inline]
    fn is_float(self) -> bool {
        self.0 >= MIN_DOUBLE
    }

    #[inline]
    fn is_number(self) -> bool {
        if self.0 >= MIN_DOUBLE || self.0 & TAG_MASK == FLAG_SMALL_INTEGER {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::BigInteger(_)) => true,
            _ => false
        }
    }

    #[inline]
    fn is_function(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_CLOSURE)
    }

    #[inline]
    fn is_tuple(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_TUPLE)
    }

    #[inline]
    fn is_map(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_MAP)
    }

    #[inline]
    fn is_local_pid(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_PID)
    }

    #[inline]
    fn is_remote_pid(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_EXTERN_PID)
    }

    #[inline]
    fn is_local_port(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_PORT)
    }

    #[inline]
    fn is_remote_port(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_EXTERN_PORT)
    }

    #[inline]
    fn is_local_reference(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_REFERENCE)
    }

    #[inline]
    fn is_remote_reference(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_EXTERN_REF)
    }

    #[inline]
    fn is_resource_reference(self) -> bool {
        !self.is_float() && (self.0 & TAG_MASK == FLAG_RESOURCE_REFERENCE)
    }

    #[inline]
    fn is_procbin(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_PROCBIN)
    }

    #[inline]
    fn is_heapbin(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_HEAPBIN)
    }

    #[inline]
    fn is_subbinary(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_SUBBINARY)
    }

    #[inline]
    fn is_match_context(self) -> bool {
        !self.is_float() && (self.0 & SUBTAG_MASK == FLAG_MATCH_CTX)
    }

    #[inline]
    fn is_boxed(self) -> bool {
        self.0 <= MAX_ADDR
    }

    #[inline]
    fn is_header(self) -> bool {
        !self.is_float() && !self.is_immediate() && !self.is_none()
    }

    #[inline]
    fn is_immediate(self) -> bool {
        match self.type_of() {
            Tag::Float => true,
            Tag::SmallInteger => true,
            Tag::Atom => true,
            Tag::Pid => true,
            Tag::Port => true,
            Tag::Nil => true,
            _ => false,
        }
    }

    #[inline]
    fn sizeof(&self) -> usize {
        if self.is_header() && !self.is_none() {
            let arity = unsafe { self.decode_header_value() };
            arity as usize + 1
        } else {
            mem::size_of::<Self>()
        }
    }
}

impl fmt::Debug for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.type_of() {
            Tag::None => write!(f, "Term(None)"),
            Tag::Nil => write!(f, "Term(Nil)"),
            Tag::List => {
                let ptr = (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm;
                let unboxed = unsafe { *ptr };
                if unboxed.is_none() {
                    let forwarding_addr_ptr = unsafe { ptr.offset(1) };
                    let forwarding_addr = unsafe { *forwarding_addr_ptr };
                    write!(f, "MoveMarker({:?} => {:?})", ptr, forwarding_addr)
                } else {
                    let value = unsafe { self.decode_list() };
                    write!(f, "Term({:?})", value)
                }
            }
            Tag::SmallInteger => {
                let value = unsafe { self.decode_smallint() };
                write!(f, "Term({})", value)
            }
            Tag::Float => {
                let value = self.decode_float();
                write!(f, "Term({})", value)
            }
            Tag::Atom => {
                let value = unsafe { self.decode_atom() };
                write!(f, "Term({})", value)
            }
            Tag::Pid => {
                let value = unsafe { self.decode_pid() };
                write!(f, "Term({})", value)
            }
            Tag::Port => {
                let value = unsafe { self.decode_port() };
                write!(f, "Term({})", value)
            }
            Tag::Box => {
                let is_literal = self.0 & FLAG_LITERAL == FLAG_LITERAL;
                let ptr = (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm;
                write!(f, "Box({:p}, literal={})", ptr, is_literal)
            }
            Tag::Unknown(invalid_tag) => {
                write!(f, "InvalidTerm(tag: {:064b})", invalid_tag)
            }
            header => {
                match self.decode_header(header, None) {
                    Ok(term) => write!(f, "Term({:?})", &term),
                    Err(err) => write!(f, "InvalidHeader(tag: {:?})", &header)
                }
            }
        }
    }
}

impl fmt::Display for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.decode() {
            Ok(term) => write!(f, "{}", term),
            Err(err) => write!(f, "{:?}", err)
        }
    }
}

impl PartialEq<RawTerm> for RawTerm {
    fn eq(&self, other: &RawTerm) -> bool {
        match (self.decode(), other.decode()) {
            (Ok(ref lhs), Ok(ref rhs)) => lhs.eq(rhs),
            (Err(_), Err(_)) => true,
            _ => false,
        }
    }
}

impl Eq for RawTerm {}

impl PartialOrd<RawTerm> for RawTerm {
    fn partial_cmp(&self, other: &RawTerm) -> Option<core::cmp::Ordering> {
        if let Ok(ref lhs) = self.decode() {
            if let Ok(ref rhs) = other.decode() {
                return lhs.partial_cmp(rhs);
            }
        }
        None
    }
}

impl Ord for RawTerm {
    fn cmp(&self, other: &RawTerm) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl core::hash::Hash for RawTerm {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.decode().unwrap().hash(state)
    }
}
