#![cfg_attr(not(target_arch = "x86_64"), allow(unused))]
use core::cmp;
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
use core::fmt;

use crate::erts::exception;

use crate::erts::term::prelude::*;

use super::{Repr, Tag};

pub type Word = u64;

const NUM_BITS: u64 = 64;

const TAG_SHIFT: u64 = 47;
const TAG_MASK: u64 = 0xFu64 << TAG_SHIFT;

const SUBTAG_SHIFT: u64 = TAG_SHIFT - 2;
const SUBTAG_MASK: u64 = 0xFCu64 << (TAG_SHIFT - 4);

const HEADER_SHIFT: u64 = 2;

// The highest allowed address in the pointer range
const MAX_ADDR: u64 = (1 << TAG_SHIFT) - 1;
// The beginning of the double range
const MIN_DOUBLE: u64 = !(i64::min_value() >> 12) as u64;
// The mask for the bits containing an immediate value
const IMMEDIATE_MASK: u64 = MAX_ADDR;

// The valid range of integer values that can fit in a term with primary tag
pub const MAX_IMMEDIATE_VALUE: u64 = MAX_ADDR;
// The valid range of integer values that can fit in a term with primary + secondary tag
pub const MAX_HEADER_VALUE: u64 = MAX_ADDR >> HEADER_SHIFT;

// The largest atom ID that can be encoded
pub const MAX_ATOM_ID: u64 = MAX_IMMEDIATE_VALUE;

// The valid range of fixed-width integers
pub const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> (NUM_BITS - TAG_SHIFT);
pub const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> (NUM_BITS - TAG_SHIFT);

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
// const FLAG_FLOAT: u64 = u64::max_value();

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
const FLAG_PROCBIN: u64 = FLAG_BINARY;
const FLAG_HEAPBIN: u64 = (1 << SUBTAG_SHIFT) | FLAG_BINARY;
const FLAG_SUBBINARY: u64 = (2 << SUBTAG_SHIFT) | FLAG_BINARY;
const FLAG_MATCH_CTX: u64 = (3 << SUBTAG_SHIFT) | FLAG_BINARY;

const FLAG_EXTERNAL: u64 = 14 << TAG_SHIFT;
const FLAG_EXTERN_PID: u64 = FLAG_EXTERNAL;
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
    fn decode_float(self) -> Float {
        debug_assert!(self.0 >= MIN_DOUBLE);
        Float::new(f64::from_bits(self.0 - MIN_DOUBLE))
    }
}
impl fmt::Binary for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#b}", self.value())
    }
}
impl Repr for RawTerm {
    type Word = u64;

    #[inline]
    fn as_usize(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn word_to_usize(word: u64) -> usize {
        word as usize
    }

    #[inline]
    fn value(&self) -> u64 {
        self.0
    }

    #[inline]
    fn type_of(&self) -> Tag<u64> {
        let term = self.0;
        if term >= MIN_DOUBLE {
            return Tag::Float;
        } else if term == 0 {
            return Tag::None;
        } else if term == FLAG_NIL {
            return Tag::Nil;
        } else if term <= MAX_ADDR {
            if term & FLAG_LITERAL == FLAG_LITERAL {
                return Tag::Literal;
            } else {
                return Tag::Box;
            }
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
                FLAG_SMALL_INTEGER => Tag::SmallInteger,
                FLAG_ATOM => Tag::Atom,
                FLAG_PID => Tag::Pid,
                FLAG_PORT => Tag::Port,
                FLAG_LIST => Tag::List,
                FLAG_TUPLE => Tag::Tuple,
                FLAG_CLOSURE => Tag::Closure,
                FLAG_BIG_INTEGER => Tag::BigInteger,
                FLAG_REFERENCE => Tag::Reference,
                FLAG_RESOURCE_REFERENCE => Tag::ResourceReference,
                FLAG_MAP => Tag::Map,
                FLAG_NONE if term == NONE => Tag::None,
                tag => Tag::Unknown(tag),
            }
        }
    }

    #[inline]
    fn encode_immediate(value: u64, tag: u64) -> Self {
        debug_assert!(tag <= TAG_MASK, "invalid primary tag: {}", tag);
        debug_assert!(tag > MAX_ADDR, "invalid primary tag: {}", tag);
        debug_assert!(value <= MAX_ADDR, "invalid immediate value: {:064b}", value);
        Self(value | tag)
    }

    #[inline]
    fn encode_header(value: u64, tag: u64) -> Self {
        debug_assert!(tag <= SUBTAG_MASK, "invalid header tag: {}", tag);
        debug_assert!(tag > MAX_HEADER_VALUE, "invalid header tag: {}", tag);
        debug_assert!(value <= MAX_HEADER_VALUE, "invalid header value: {}", value);
        Self(value | tag)
    }

    #[inline]
    fn encode_list(ptr: *const Cons) -> Self {
        let value = ptr as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        Self(value | FLAG_LIST)
    }

    #[inline]
    fn encode_box<T>(ptr: *const T) -> Self
    where
        T: ?Sized,
    {
        let value = ptr as *const () as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        Self(value | FLAG_BOXED)
    }

    #[inline]
    fn encode_literal<T>(ptr: *const T) -> Self
    where
        T: ?Sized,
    {
        let value = ptr as *const () as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        Self(value | FLAG_LITERAL | FLAG_BOXED)
    }

    #[inline]
    unsafe fn decode_box(self) -> *mut Self {
        (self.0 & !(TAG_MASK | FLAG_LITERAL)) as *const RawTerm as *mut RawTerm
    }

    #[inline]
    unsafe fn decode_list(self) -> Boxed<Cons> {
        debug_assert_eq!(self.0 & TAG_MASK, FLAG_LIST);
        let ptr = (self.0 & !TAG_MASK) as *const Cons as *mut Cons;
        Boxed::new_unchecked(ptr)
    }

    #[inline]
    unsafe fn decode_smallint(self) -> SmallInteger {
        const SMALL_INTEGER_SIGNED: u64 = 1u64 << (TAG_SHIFT - 1);

        let value = self.0 & !TAG_MASK;
        let i = if value & SMALL_INTEGER_SIGNED == SMALL_INTEGER_SIGNED {
            !MAX_ADDR | value
        } else {
            MAX_ADDR & value
        } as i64;
        SmallInteger::new_unchecked(i as isize)
    }

    #[inline]
    unsafe fn decode_immediate(self) -> u64 {
        self.0 & IMMEDIATE_MASK
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
        debug_assert!(
            self.0 < MIN_DOUBLE,
            "invalid use of immediate float value as header"
        );
        debug_assert!(self.0 > MAX_ADDR, "invalid use of boxed value as header");
        self.0 & MAX_HEADER_VALUE
    }
}

unsafe impl Send for RawTerm {}

impl Encode<RawTerm> for u8 {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm::encode_immediate(
            (*self) as u64,
            FLAG_SMALL_INTEGER,
        ))
    }
}

impl Encode<RawTerm> for SmallInteger {
    fn encode(&self) -> exception::Result<RawTerm> {
        let i: i64 = (*self).into();
        Ok(RawTerm::encode_immediate(
            (i as u64) & MAX_ADDR,
            FLAG_SMALL_INTEGER,
        ))
    }
}

impl Encode<RawTerm> for Float {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm(self.value().to_bits() + MIN_DOUBLE))
    }
}

impl Encode<RawTerm> for bool {
    fn encode(&self) -> exception::Result<RawTerm> {
        let atom: Atom = (*self).into();
        Ok(RawTerm::encode_immediate(atom.id() as u64, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Atom {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.id() as u64, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Pid {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.as_usize() as u64, FLAG_PID))
    }
}

impl Encode<RawTerm> for Port {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.as_usize() as u64, FLAG_PORT))
    }
}

impl From<*mut RawTerm> for RawTerm {
    fn from(ptr: *mut RawTerm) -> Self {
        RawTerm::encode_box(ptr)
    }
}

impl From<Float> for RawTerm {
    fn from(f: Float) -> Self {
        f.encode().unwrap()
    }
}

impl_list!(RawTerm);
impl_boxable!(BigInteger, RawTerm);
impl_boxable!(Reference, RawTerm);
impl_boxable!(ExternalPid, RawTerm);
impl_boxable!(ExternalPort, RawTerm);
impl_boxable!(ExternalReference, RawTerm);
impl_boxable!(Resource, RawTerm);
impl_boxable!(Map, RawTerm);
impl_boxable!(ProcBin, RawTerm);
impl_boxable!(SubBinary, RawTerm);
impl_boxable!(MatchContext, RawTerm);
impl_unsized_boxable!(Tuple, RawTerm);
impl_unsized_boxable!(Closure, RawTerm);
impl_unsized_boxable!(HeapBin, RawTerm);
impl_literal!(BinaryLiteral, RawTerm);

impl Cast<*mut RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *mut RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_non_empty_list());
        unsafe { self.decode_box() }
    }
}

impl<T> Cast<Boxed<T>> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> Boxed<T> {
        assert!(self.is_boxed() || self.is_literal() || self.is_non_empty_list());
        Boxed::new(unsafe { self.decode_box() as *mut T }).unwrap()
    }
}

impl<T> Cast<*mut T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *mut T {
        assert!(self.is_boxed() || self.is_literal());
        unsafe { self.decode_box() as *mut T }
    }
}

impl Cast<*mut Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *mut Cons {
        assert!(self.is_non_empty_list());
        unsafe { self.decode_box() as *mut Cons }
    }
}

impl Cast<*const RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *const RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_non_empty_list());
        unsafe { self.decode_box() as *const RawTerm }
    }
}

impl<T> Cast<*const T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *const T {
        assert!(self.is_boxed() || self.is_literal());
        unsafe { self.decode_box() as *const T }
    }
}

impl Cast<*const Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *const Cons {
        assert!(self.is_non_empty_list());
        unsafe { self.decode_box() as *const Cons }
    }
}

impl Encoded for RawTerm {
    #[inline]
    fn decode(&self) -> exception::Result<TypedTerm> {
        let tag = self.type_of();
        match tag {
            Tag::Nil => Ok(TypedTerm::Nil),
            Tag::List => Ok(TypedTerm::List(unsafe { self.decode_list() })),
            Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe { self.decode_smallint() })),
            // When compiling for non-x86_64, we use boxed floats, so
            // we accomodate that by boxing the float.
            #[cfg(all(target_pointer_width = "64", target_arch = "x86_64"))]
            Tag::Float => Ok(TypedTerm::Float(self.decode_float())),
            #[cfg(not(all(target_pointer_width = "64", target_arch = "x86_64")))]
            Tag::Float => Ok(TypedTerm::Float(unsafe {
                Boxed::new_unchecked(self as *const _ as *mut Float)
            })),
            Tag::Atom => Ok(TypedTerm::Atom(unsafe { self.decode_atom() })),
            Tag::Pid => Ok(TypedTerm::Pid(unsafe { self.decode_pid() })),
            Tag::Port => Ok(TypedTerm::Port(unsafe { self.decode_port() })),
            Tag::Box => {
                let ptr = unsafe { self.decode_box() };
                let unboxed = unsafe { &*ptr };
                match unboxed.type_of() {
                    Tag::Nil => Ok(TypedTerm::Nil),
                    Tag::List => Ok(TypedTerm::List(unsafe { unboxed.decode_list() })),
                    Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe {
                        unboxed.decode_smallint()
                    })),
                    #[cfg(all(target_pointer_width = "64", target_arch = "x86_64"))]
                    Tag::Float => Ok(TypedTerm::Float(unboxed.decode_float())),
                    #[cfg(not(all(target_pointer_width = "64", target_arch = "x86_64")))]
                    Tag::Float => Ok(TypedTerm::Float(unsafe {
                        Boxed::new_unchecked(ptr as *mut Float)
                    })),
                    Tag::Atom => Ok(TypedTerm::Atom(unsafe { unboxed.decode_atom() })),
                    Tag::Pid => Ok(TypedTerm::Pid(unsafe { unboxed.decode_pid() })),
                    Tag::Port => Ok(TypedTerm::Port(unsafe { unboxed.decode_port() })),
                    Tag::Box => Err(TermDecodingError::MoveMarker.into()),
                    Tag::Unknown(_) => Err(TermDecodingError::InvalidTag.into()),
                    Tag::None => Err(TermDecodingError::NoneValue.into()),
                    header => unboxed.decode_header(header, Some(self.is_literal())),
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
        self.0 <= MAX_ADDR && self.0 & FLAG_LITERAL == FLAG_LITERAL && (self.0 & !FLAG_LITERAL > 0)
    }

    #[inline]
    fn is_list(self) -> bool {
        !self.is_float() && ((self.0 == FLAG_NIL) || (self.0 & TAG_MASK == FLAG_LIST))
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
            _ => false,
        }
    }

    #[inline]
    fn is_float(self) -> bool {
        self.0 >= MIN_DOUBLE
    }

    #[inline]
    fn is_boxed_float(self) -> bool {
        self.is_float()
    }

    #[inline]
    fn is_number(self) -> bool {
        if self.0 >= MIN_DOUBLE || self.0 & TAG_MASK == FLAG_SMALL_INTEGER {
            return true;
        }
        match self.decode() {
            Ok(TypedTerm::BigInteger(_)) => true,
            _ => false,
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
        // >1 means it is not null or a null literal pointer
        self.0 <= MAX_ADDR && self.0 > 1
    }

    #[inline]
    fn is_header(self) -> bool {
        // All headers have a tag + optional subtag, and all
        // header tags begin at FLAG_TUPLE and go up to the highest
        // tag value.
        !self.is_float() && (self.0 & SUBTAG_MASK) >= FLAG_TUPLE
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
}

impl fmt::Debug for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.type_of() {
            Tag::None => write!(f, "Term(None)"),
            Tag::Nil => write!(f, "Term(Nil)"),
            Tag::List => {
                let ptr = unsafe { self.decode_box() };
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
                let ptr = unsafe { self.decode_box() };
                let unboxed = unsafe { &*ptr };
                write!(
                    f,
                    "Box({:p}, literal={}, value={:?})",
                    ptr, is_literal, unboxed
                )
            }
            Tag::Unknown(invalid_tag) => write!(f, "InvalidTerm(tag: {:064b})", invalid_tag),
            header => match self.decode_header(header, None) {
                Ok(term) => write!(f, "Term({:?})", &term),
                Err(_) => write!(f, "InvalidHeader(tag: {:?})", &header),
            },
        }
    }
}

impl fmt::Display for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.decode() {
            Ok(term) => write!(f, "{}", term),
            Err(err) => write!(f, "{:?}", err),
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
                dbg!(lhs);
                dbg!(rhs);
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

#[cfg(all(test, target_pointer_width = "64", target_arch = "x86_64"))]
mod tests {
    use core::convert::TryInto;

    use crate::borrow::CloneToProcess;
    use crate::erts::process::alloc::TermAlloc;
    use crate::erts::testing::RegionHeap;

    use super::*;

    #[test]
    fn none_encoding_x86_64() {
        assert_eq!(RawTerm::NONE, RawTerm::NONE);
        assert!(RawTerm::NONE.is_none());
        assert_eq!(RawTerm::NONE.type_of(), Tag::None);
        assert!(!RawTerm::NONE.is_boxed());
        assert!(!RawTerm::NONE.is_header());
        assert!(!RawTerm::NONE.is_immediate());

        let none: *const BigInteger = core::ptr::null();
        let none_boxed: RawTerm = none.into();
        assert!(none_boxed.is_none());
        assert_eq!(none_boxed.type_of(), Tag::None);
        assert!(!none_boxed.is_boxed());
        assert!(!none_boxed.is_bigint());
    }

    #[test]
    fn literal_encoding_x86_64() {
        let literal: *const BigInteger = core::ptr::null();
        let literal_boxed = RawTerm::encode_literal(literal);

        assert!(!literal_boxed.is_boxed());
        assert!(!literal_boxed.is_literal());
        assert_eq!(literal_boxed.type_of(), Tag::Literal);
        assert!(!literal_boxed.is_header());
        assert!(!literal_boxed.is_immediate());

        let literal: *const BigInteger = 0xABCDusize as *const usize as *const BigInteger;
        let literal_boxed = RawTerm::encode_literal(literal);

        assert!(literal_boxed.is_boxed());
        assert!(literal_boxed.is_literal());
        assert_eq!(literal_boxed.type_of(), Tag::Literal);
        assert!(!literal_boxed.is_header());
        assert!(!literal_boxed.is_immediate());
    }

    #[test]
    fn float_encoding_x86_64() {
        let float: Float = std::f64::MAX.into();

        let float_term: RawTerm = float.encode().unwrap();
        assert!(float_term.is_float());
        assert_eq!(float_term.type_of(), Tag::Float);
        assert!(float_term.is_immediate());
        assert!(!float_term.is_header());
        assert!(!float_term.is_boxed());

        let float_decoded: Result<Float, _> = float_term.decode().unwrap().try_into();
        assert!(float_decoded.is_ok());
        assert_eq!(float, float_decoded.unwrap());

        let nan = RawTerm(std::f64::NAN.to_bits());
        assert!(nan.is_float());
        assert_eq!(nan.type_of(), Tag::Float);
        assert!(nan.is_immediate());
        assert!(!nan.is_header());
        assert!(!nan.is_boxed());
    }

    #[test]
    fn fixnum_encoding_x86_64() {
        let max: SmallInteger = MAX_SMALLINT_VALUE.try_into().unwrap();
        let min: SmallInteger = MIN_SMALLINT_VALUE.try_into().unwrap();

        let max_term: RawTerm = max.encode().unwrap();
        let min_term: RawTerm = min.encode().unwrap();
        assert!(max_term.is_integer());
        assert!(min_term.is_integer());
        assert_eq!(max_term.type_of(), Tag::SmallInteger);
        assert_eq!(min_term.type_of(), Tag::SmallInteger);
        assert!(max_term.is_smallint());
        assert!(min_term.is_smallint());
        assert!(max_term.is_immediate());
        assert!(min_term.is_immediate());
        assert!(!max_term.is_header());
        assert!(!min_term.is_header());
        assert!(!max_term.is_boxed());
        assert!(!min_term.is_boxed());

        let max_decoded: Result<SmallInteger, _> = max_term.decode().unwrap().try_into();
        assert!(max_decoded.is_ok());
        assert_eq!(max, max_decoded.unwrap());

        let min_decoded: Result<SmallInteger, _> = min_term.decode().unwrap().try_into();
        assert!(min_decoded.is_ok());
        assert_eq!(min, min_decoded.unwrap());

        let a: SmallInteger = 101usize.try_into().unwrap();
        let a_term: RawTerm = a.encode().unwrap();
        assert!(a_term.is_integer());
        assert_eq!(a_term.type_of(), Tag::SmallInteger);
        assert!(a_term.is_smallint());
        assert!(!a_term.is_header());
        assert!(!a_term.is_boxed());
        assert_eq!(a_term.arity(), 0);
    }

    #[test]
    fn atom_encoding_x86_64() {
        let atom = unsafe { Atom::from_id(MAX_ATOM_ID as usize) };

        let atom_term: RawTerm = atom.encode().unwrap();
        assert_eq!(atom_term.type_of(), Tag::Atom);
        assert!(atom_term.is_atom());
        assert!(atom_term.is_immediate());
        assert!(!atom_term.is_integer());
        assert!(!atom_term.is_header());
        assert!(!atom_term.is_boxed());

        let atom_decoded: Result<Atom, _> = atom_term.decode().unwrap().try_into();
        assert!(atom_decoded.is_ok());
        assert_eq!(atom, atom_decoded.unwrap());
    }

    #[test]
    fn pid_encoding_x86_64() {
        let pid = unsafe { Pid::from_raw(MAX_ADDR as usize) };

        let pid_term: RawTerm = pid.encode().unwrap();
        assert!(pid_term.is_local_pid());
        assert!(!pid_term.is_remote_pid());
        assert_eq!(pid_term.type_of(), Tag::Pid);
        assert!(pid_term.is_immediate());
        assert!(!pid_term.is_integer());
        assert!(!pid_term.is_header());
        assert!(!pid_term.is_boxed());

        let pid_decoded: Result<Pid, _> = pid_term.decode().unwrap().try_into();
        assert!(pid_decoded.is_ok());
        assert_eq!(pid, pid_decoded.unwrap());

        // This function pierces boxes
        assert!(pid_term.is_pid());
    }

    #[test]
    fn port_encoding_x86_64() {
        let port = unsafe { Port::from_raw(MAX_ADDR as usize) };

        let port_term: RawTerm = port.encode().unwrap();
        assert!(port_term.is_local_port());
        assert!(!port_term.is_remote_port());
        assert_eq!(port_term.type_of(), Tag::Port);
        assert!(port_term.is_immediate());
        assert!(!port_term.is_integer());
        assert!(!port_term.is_header());
        assert!(!port_term.is_boxed());

        let port_decoded: Result<Port, _> = port_term.decode().unwrap().try_into();
        assert!(port_decoded.is_ok());
        assert_eq!(port, port_decoded.unwrap());

        // This function pierces boxes
        assert!(port_term.is_port());
    }

    #[test]
    fn bigint_encoding_x86_64() {
        let big: BigInteger = (MAX_SMALLINT_VALUE + 1).try_into().unwrap();
        let boxed = Boxed::new(&big as *const _ as *mut BigInteger).unwrap();

        let big_term: RawTerm = boxed.encode().unwrap();
        assert!(big_term.is_boxed());
        assert_eq!(big_term.type_of(), Tag::Box);
        assert!(!big_term.is_bigint());

        let unboxed: *const RawTerm = big_term.dyn_cast();
        let big_header = unsafe { *unboxed };
        assert!(big_header.is_header());
        assert!(big_header.is_bigint());
        assert_eq!(big_header.type_of(), Tag::BigInteger);

        let big_decoded: Result<Boxed<BigInteger>, _> = big_term.decode().unwrap().try_into();
        assert!(big_decoded.is_ok());
        assert_eq!(&big, big_decoded.unwrap().as_ref());
    }

    #[test]
    fn tuple_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        // Empty tuple
        let tuple = Tuple::new(&mut heap, 0).unwrap();
        let tuple_term: RawTerm = tuple.encode().unwrap();
        assert!(tuple_term.is_boxed());
        assert_eq!(tuple_term.type_of(), Tag::Box);
        assert!(!tuple_term.is_tuple());

        let unboxed: *const RawTerm = tuple_term.dyn_cast();
        let tuple_header = unsafe { *unboxed };
        assert!(tuple_header.is_header());
        assert!(tuple_header.is_tuple());
        assert_eq!(tuple_header.type_of(), Tag::Tuple);

        let tuple_decoded: Result<Boxed<Tuple>, _> = tuple_term.decode().unwrap().try_into();
        assert!(tuple_decoded.is_ok());
        let tuple_box = tuple_decoded.unwrap();
        assert_eq!(&tuple, tuple_box.as_ref());
        assert_eq!(tuple_box.len(), 0);

        // Non-empty tuple
        let elements = vec![fixnum!(1), fixnum!(2), fixnum!(3), fixnum!(4)];
        let tuple2 = Tuple::from_slice(&mut heap, elements.as_slice()).unwrap();
        let tuple2_term: RawTerm = tuple2.encode().unwrap();
        assert!(tuple2_term.is_boxed());
        assert_eq!(tuple2_term.type_of(), Tag::Box);
        assert!(!tuple2_term.is_tuple());

        let unboxed: *const RawTerm = tuple2_term.dyn_cast();
        let tuple2_header = unsafe { *unboxed };
        assert!(tuple2_header.is_header());
        assert!(tuple2_header.is_tuple());
        assert_eq!(tuple2_header.type_of(), Tag::Tuple);

        let tuple2_decoded: Result<Boxed<Tuple>, _> = tuple2_term.decode().unwrap().try_into();
        assert!(tuple2_decoded.is_ok());
        let tuple2_box = tuple2_decoded.unwrap();
        assert_eq!(&tuple2, tuple2_box.as_ref());
        assert_eq!(tuple2_box.len(), 4);
        assert_eq!(tuple2_box.get_element(0), Ok(fixnum!(1)));
        assert_eq!(tuple2_box.get_element(3), Ok(fixnum!(4)));
    }

    #[test]
    fn list_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        // Empty list
        assert!(RawTerm::NIL.is_list());
        assert!(!RawTerm::NIL.is_non_empty_list());
        assert_eq!(RawTerm::NIL.type_of(), Tag::Nil);
        assert!(RawTerm::NIL.is_nil());
        assert!(RawTerm::NIL.is_immediate());

        // Non-empty list
        let list = cons!(heap, fixnum!(1), fixnum!(2));
        let list_term: RawTerm = list.encode().unwrap();
        assert!(!list_term.is_boxed());
        assert!(list_term.is_non_empty_list());
        assert_eq!(list_term.type_of(), Tag::List);

        let unboxed: *const RawTerm = list_term.dyn_cast();
        let car = unsafe { *unboxed };
        assert!(!car.is_header());
        assert!(car.is_smallint());
        assert_eq!(car.type_of(), Tag::SmallInteger);

        let list_decoded: Result<Boxed<Cons>, _> = list_term.decode().unwrap().try_into();
        assert!(list_decoded.is_ok());
        let list_box = list_decoded.unwrap();
        assert_eq!(&list, list_box.as_ref());
        assert_eq!(list_box.count(), Some(2));
    }

    #[test]
    fn map_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        let pairs = vec![(atom!("foo"), fixnum!(1)), (atom!("bar"), fixnum!(2))];
        let map = Map::from_slice(pairs.as_slice());
        let map_term = map.clone_to_heap(&mut heap).unwrap();
        assert!(map_term.is_boxed());
        assert_eq!(map_term.type_of(), Tag::Box);
        assert!(!map_term.is_map());

        let unboxed: *const RawTerm = map_term.dyn_cast();
        let map_header = unsafe { *unboxed };
        assert!(map_header.is_header());
        assert!(map_header.is_map());
        assert_eq!(map_header.type_of(), Tag::Map);

        let map_decoded: Result<Boxed<Map>, _> = map_term.decode().unwrap().try_into();
        assert!(map_decoded.is_ok());
        let map_box = map_decoded.unwrap();
        assert_eq!(&map, map_box.as_ref());
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(atom!("bar")), Some(fixnum!(2)));
    }

    #[test]
    fn closure_encoding_x86_64() {
        use crate::erts::process::Process;
        use crate::erts::term::closure::*;
        use alloc::sync::Arc;

        let mut heap = RegionHeap::default();
        let creator = Pid::new(1, 0).unwrap();

        let module = Atom::try_from_str("module").unwrap();
        let arity = 0;
        let code = |_arc_process: &Arc<Process>| Ok(());

        let one = fixnum!(1);
        let two = fixnum!(2);
        let index = 1 as Index;
        let old_unique = 2 as OldUnique;
        let unique = [0u8; 16];
        let closure = heap
            .anonymous_closure_with_env_from_slices(
                module,
                index,
                old_unique,
                unique,
                arity,
                Some(code),
                Creator::Local(creator),
                &[&[one, two]],
            )
            .unwrap();
        let mfa = closure.module_function_arity();
        assert_eq!(closure.env_len(), 2);
        let closure_term: RawTerm = closure.into();
        assert!(closure_term.is_boxed());
        assert_eq!(closure_term.type_of(), Tag::Box);
        assert!(!closure_term.is_function());

        let unboxed: *const RawTerm = closure_term.dyn_cast();
        let closure_header = unsafe { *unboxed };
        assert!(closure_header.is_header());
        assert!(closure_header.is_function());
        assert_eq!(closure_header.type_of(), Tag::Closure);

        let closure_decoded: Result<Boxed<Closure>, _> = closure_term.decode().unwrap().try_into();
        assert!(closure_decoded.is_ok());
        let closure_box = closure_decoded.unwrap();
        assert_eq!(&closure, closure_box.as_ref());
        assert_eq!(closure_box.env_len(), 2);
        assert_eq!(closure_box.module_function_arity(), mfa);
        assert_eq!(closure_box.get_env_element(0), one);
        assert_eq!(closure_box.get_env_element(1), two);
    }

    #[test]
    fn procbin_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        let bin = heap.procbin_from_str("hello world!").unwrap();
        assert_eq!(bin.as_str(), "hello world!");
        let bin_term: RawTerm = bin.into();
        assert!(bin_term.is_boxed());
        assert_eq!(bin_term.type_of(), Tag::Box);
        assert!(!bin_term.is_procbin());

        let unboxed: *const RawTerm = bin_term.dyn_cast();
        let bin_header = unsafe { *unboxed };
        assert!(bin_header.is_header());
        assert!(bin_header.is_procbin());
        assert_eq!(bin_header.type_of(), Tag::ProcBin);

        let bin_decoded: Result<Boxed<ProcBin>, _> = bin_term.decode().unwrap().try_into();
        assert!(bin_decoded.is_ok());
        let bin_box = bin_decoded.unwrap();
        assert_eq!(&bin, bin_box.as_ref());
        assert_eq!(bin_box.as_str(), "hello world!");

        // These functions pierce the box
        assert!(bin_term.is_binary());
        assert!(bin_term.is_bitstring());
    }

    #[test]
    fn heapbin_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        let bin = heap.heapbin_from_str("hello world!").unwrap();
        assert_eq!(bin.as_str(), "hello world!");
        let bin_term: RawTerm = bin.into();
        assert!(bin_term.is_boxed());
        assert_eq!(bin_term.type_of(), Tag::Box);
        assert!(!bin_term.is_procbin());

        let unboxed: *const RawTerm = bin_term.dyn_cast();
        let bin_header = unsafe { *unboxed };
        assert!(bin_header.is_header());
        assert!(bin_header.is_heapbin());
        assert_eq!(bin_header.type_of(), Tag::HeapBinary);

        let bin_decoded: Result<Boxed<HeapBin>, _> = bin_term.decode().unwrap().try_into();
        assert!(bin_decoded.is_ok());
        let bin_box = bin_decoded.unwrap();
        assert_eq!(&bin, bin_box.as_ref());
        //panic!("ok");
        assert_eq!(bin_box.as_str(), "hello world!");

        // These functions pierce the box
        assert!(bin_term.is_binary());
        assert!(bin_term.is_bitstring());
    }

    #[test]
    fn subbinary_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        let bin = heap.heapbin_from_str("hello world!").unwrap();
        let bin_term: RawTerm = bin.into();
        // Slice out 'world!'
        let byte_offset = 6;
        let len = 6;
        let sub = heap
            .subbinary_from_original(bin_term, byte_offset, 0, len, 0)
            .unwrap();
        let sub_term: RawTerm = sub.into();

        assert!(sub_term.is_boxed());
        assert_eq!(sub_term.type_of(), Tag::Box);
        assert!(!sub_term.is_subbinary());

        let unboxed: *const RawTerm = sub_term.dyn_cast();
        let sub_header = unsafe { *unboxed };

        assert!(sub_header.is_header());
        assert!(sub_header.is_subbinary());
        assert_eq!(sub_header.type_of(), Tag::SubBinary);

        let sub_decoded: Result<Boxed<SubBinary>, _> = sub_term.decode().unwrap().try_into();
        assert!(sub_decoded.is_ok());
        let sub_box = sub_decoded.unwrap();
        assert_eq!(&sub, sub_box.as_ref());
        assert!(sub_box.is_aligned());
        assert!(sub_box.is_binary());
        assert_eq!(sub_box.try_into(), Ok("world!".to_owned()));
    }

    #[test]
    fn match_context_encoding_x86_64() {
        let mut heap = RegionHeap::default();

        let bin = heap.heapbin_from_str("hello world!").unwrap();
        let match_ctx = heap.match_context_from_binary(bin).unwrap();
        let match_ctx_term: RawTerm = match_ctx.into();

        assert!(match_ctx_term.is_boxed());
        assert_eq!(match_ctx_term.type_of(), Tag::Box);
        assert!(!match_ctx_term.is_match_context());

        let unboxed: *const RawTerm = match_ctx_term.dyn_cast();
        let match_ctx_header = unsafe { *unboxed };
        assert!(match_ctx_header.is_header());
        assert!(match_ctx_header.is_match_context());
        assert_eq!(match_ctx_header.type_of(), Tag::MatchContext);

        let match_ctx_decoded: Result<Boxed<MatchContext>, _> =
            match_ctx_term.decode().unwrap().try_into();
        assert!(match_ctx_decoded.is_ok());
        let match_ctx_box = match_ctx_decoded.unwrap();
        assert_eq!(&match_ctx, match_ctx_box.as_ref());
        assert!(match_ctx_box.is_aligned());
        assert!(match_ctx_box.is_binary());
        assert_eq!(match_ctx_box.try_into(), Ok("hello world!".to_owned()));
    }

    #[test]
    fn resource_encoding_x86_64() {
        use core::any::Any;

        let mut heap = RegionHeap::default();

        // Need a concrete type for casting
        let code: Box<dyn Any> = Box::new(Predicate::new(|input: bool| Some(input)));
        let resource = Resource::from_value(&mut heap, code).unwrap();
        let resource_term: RawTerm = resource.into();
        assert!(resource_term.is_boxed());
        assert_eq!(resource_term.type_of(), Tag::Box);
        assert!(!resource_term.is_resource_reference());

        let unboxed: *const RawTerm = resource_term.dyn_cast();
        let resource_header = unsafe { *unboxed };
        assert!(resource_header.is_header());
        assert!(resource_header.is_resource_reference());
        assert_eq!(resource_header.type_of(), Tag::ResourceReference);

        let resource_decoded: Result<Boxed<Resource>, _> =
            resource_term.decode().unwrap().try_into();
        assert!(resource_decoded.is_ok());
        let resource_box = resource_decoded.unwrap();
        assert_eq!(&resource, resource_box.as_ref());
        let resource_code = resource.downcast_ref::<Predicate>().unwrap();
        assert_eq!(resource_code.invoke(true), Some(true));
    }

    #[test]
    fn reference_encoding_x86_64() {
        use crate::erts::scheduler;
        let mut heap = RegionHeap::default();

        let reference = heap.reference(scheduler::id::next(), 0).unwrap();
        let reference_term: RawTerm = reference.into();
        assert!(reference_term.is_boxed());
        assert_eq!(reference_term.type_of(), Tag::Box);
        assert!(!reference_term.is_local_reference());

        let unboxed: *const RawTerm = reference_term.dyn_cast();
        let reference_header = unsafe { *unboxed };
        assert!(reference_header.is_header());
        assert!(reference_header.is_local_reference());
        assert_eq!(reference_header.type_of(), Tag::Reference);

        let reference_decoded: Result<Boxed<Reference>, _> =
            reference_term.decode().unwrap().try_into();
        assert!(reference_decoded.is_ok());
        let reference_box = reference_decoded.unwrap();
        assert_eq!(&reference, reference_box.as_ref());

        // This function pierces the box
        assert!(reference_term.is_reference());
    }

    #[test]
    fn external_pid_encoding_x86_64() {
        use crate::erts::Node;
        use alloc::sync::Arc;

        let mut heap = RegionHeap::default();

        let arc_node = Arc::new(Node::new(
            1,
            Atom::try_from_str("node@external").unwrap(),
            0,
        ));
        let pid = ExternalPid::new(arc_node, 1, 0).unwrap();
        let pid_term = pid.clone_to_heap(&mut heap).unwrap();
        assert!(pid_term.is_boxed());
        assert_eq!(pid_term.type_of(), Tag::Box);
        assert!(!pid_term.is_remote_pid());

        let unboxed: *const RawTerm = pid_term.dyn_cast();
        let pid_header = unsafe { *unboxed };
        assert!(pid_header.is_header());
        assert!(pid_header.is_remote_pid());
        assert!(!pid_header.is_local_pid());
        assert_eq!(pid_header.type_of(), Tag::ExternalPid);

        let pid_decoded: Result<Boxed<ExternalPid>, _> = pid_term.decode().unwrap().try_into();
        assert!(pid_decoded.is_ok());
        let pid_box = pid_decoded.unwrap();
        assert_eq!(&pid, pid_box.as_ref());

        // This function pierces the box
        assert!(pid_term.is_pid());
    }

    #[test]
    #[ignore]
    fn external_port_encoding_x86_64() {
        // TODO: let mut heap = RegionHeap::default();
        // Waiting on implementation of this type
    }

    #[test]
    #[ignore]
    fn external_reference_encoding_x86_64() {
        // TODO: let mut heap = RegionHeap::default();
        // Waiting on implementation of this type
    }

    struct Predicate {
        pred: Box<dyn Fn(bool) -> Option<bool>>,
    }
    impl Predicate {
        pub(super) fn new(pred: impl Fn(bool) -> Option<bool> + 'static) -> Self {
            Self {
                pred: Box::new(pred),
            }
        }

        pub(super) fn invoke(&self, input: bool) -> Option<bool> {
            (self.pred)(input)
        }
    }
}
