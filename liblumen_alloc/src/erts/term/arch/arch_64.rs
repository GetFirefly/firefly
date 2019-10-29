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

use crate::erts::exception;

use liblumen_core::sys::sysconf::MIN_ALIGN;
const_assert!(MIN_ALIGN >= 8);

use crate::erts::term::prelude::*;

use super::{Tag, Repr};

#[cfg_attr(target_arch = "x86_64", allow(unused))]
pub type Word = u64;

// The valid range of integer values that can fit in a term with primary tag
#[cfg_attr(target_arch = "x86_64", allow(unused))]
pub const MAX_IMMEDIATE_VALUE: u64 = u64::max_value() >> 3;
#[cfg_attr(target_arch = "x86_64", allow(unused))]
pub const MAX_ATOM_ID: u64 = MAX_IMMEDIATE_VALUE;

// The valid range of fixed-width integers
#[cfg_attr(target_arch = "x86_64", allow(unused))]
pub const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> 4;
#[cfg_attr(target_arch = "x86_64", allow(unused))]
pub const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> 4;

const PRIMARY_SHIFT: u64 = 3;
const HEADER_SHIFT: u64 = 8;
const HEADER_TAG_SHIFT: u64 = 3;

// Primary tags (use lowest 3 bits, since minimum alignment is 8)
const FLAG_HEADER: u64 = 0;        // 0b000
const FLAG_BOXED: u64 = 1;         // 0b001
const FLAG_LIST: u64 = 2;          // 0b010
const FLAG_LITERAL: u64 = 3;       // 0b011
const FLAG_SMALL_INTEGER: u64 = 4; // 0b100
const FLAG_ATOM: u64 = 5;          // 0b101
const FLAG_PID: u64 = 6;           // 0b110
const FLAG_PORT: u64 = 7;          // 0b111

// Header tags (uses an additional 5 bits beyond the primary tag)
// NONE is a special case where all bits of the header are zero
const FLAG_NONE: u64 = 0;                                                   // 0b00000_000
const FLAG_TUPLE: u64 = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER;              // 0b00001_000
const FLAG_BIG_INTEGER: u64 = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER;        // 0b00010_000
#[allow(unused)]
const FLAG_UNUSED: u64 = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER;             // 0b00011_000
const FLAG_REFERENCE: u64 = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER;          // 0b00100_000
const FLAG_CLOSURE: u64 = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER;            // 0b00101_000
const FLAG_FLOAT: u64 = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER;              // 0b00110_000
const FLAG_RESOURCE_REFERENCE: u64 = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00111_000
const FLAG_PROCBIN: u64 = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER;            // 0b01000_000
const FLAG_HEAPBIN: u64 = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER;            // 0b01001_000
const FLAG_SUBBINARY: u64 = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER;         // 0b01010_000
const FLAG_MATCH_CTX: u64 = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER;         // 0b01011_000
const FLAG_EXTERN_PID: u64 = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER;        // 0b01100_000
const FLAG_EXTERN_PORT: u64 = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER;       // 0b01101_000
const FLAG_EXTERN_REF: u64 = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER;        // 0b01110_000
const FLAG_MAP: u64 = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER;               // 0b01111_000
const FLAG_NIL: u64 = (16 << HEADER_TAG_SHIFT) | FLAG_HEADER;               // 0b10000_000

// The primary tag is given by masking bits 1-3
const MASK_PRIMARY: u64 = 0b111;
// Header is composed of 3 primary tag bits, and 4 subtag bits
const MASK_HEADER: u64 = 0b11111_111;

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
    pub const HEADER_FLOAT: u64 = FLAG_FLOAT;
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
}
impl fmt::Binary for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#b}", self.value())
    }
}
impl Repr for RawTerm {
    type Word = u64;

    #[inline]
    fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn value(&self) -> u64 {
        self.0
    }

    #[inline]
    fn type_of(self) -> Tag<u64> {
        let term = self.0;
        let tag = match term & MASK_PRIMARY {
            FLAG_HEADER => term & MASK_HEADER,
            tag => tag
        };

        match tag {
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
            _ => Tag::Unknown(tag)
        }
    }

    #[inline]
    fn encode_immediate(value: u64, tag: u64) -> Self {
        debug_assert!(tag <= MASK_PRIMARY, "invalid primary tag");
        Self((value << PRIMARY_SHIFT) | tag)
    }

    #[inline]
    fn encode_header(value: u64, tag: u64) -> Self {
        Self((value << HEADER_SHIFT) | tag)
    }

    #[inline]
    fn encode_list(value: *const Cons) -> Self {
        Self(value as u64 | FLAG_LIST)
    }

    #[inline]
    fn encode_box<T>(value: *const T) -> Self where T: ?Sized {
        Self(value as *const () as u64 | FLAG_BOXED)
    }

    #[inline]
    fn encode_literal<T>(value: *const T) -> Self where T: ?Sized {
        Self(value as *const () as u64 | FLAG_LITERAL)
    }

    #[inline]
    unsafe fn decode_list(self) -> Boxed<Cons> {
        debug_assert_eq!(self.0 & MASK_PRIMARY, FLAG_LIST);
        let ptr = (self.0 & !MASK_PRIMARY) as *const Cons as *mut Cons;
        Boxed::new_unchecked(ptr)
    }

    #[inline]
    unsafe fn decode_smallint(self) -> SmallInteger {
        let unmasked = (self.0 & !MASK_PRIMARY) as i64;
        let i = unmasked >> 3;
        SmallInteger::new_unchecked(i as isize)
    }

    #[inline]
    unsafe fn decode_immediate(self) -> u64 {
        (self.0 & !MASK_PRIMARY) >> PRIMARY_SHIFT
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
        debug_assert_eq!(self.0 & MASK_HEADER, 0);
        self.0 >> HEADER_SHIFT
    }
}

unsafe impl Send for RawTerm {}

impl Encode<RawTerm> for u8 {
    fn encode(&self) -> exception::Result<RawTerm> {
        Ok(RawTerm::encode_immediate((*self) as u64, FLAG_SMALL_INTEGER))
    }
}

impl Encode<RawTerm> for SmallInteger {
    fn encode(&self) -> exception::Result<RawTerm> {
        let i: i64 = (*self).into();
        Ok(RawTerm::encode_immediate(i as u64, FLAG_SMALL_INTEGER))
    }
}

impl Encode<RawTerm> for bool {
    fn encode(&self) -> exception::Result<RawTerm> {
        let atom = Atom::try_from_str(&self.to_string()).unwrap();
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

impl_list!(RawTerm);
impl_boxable!(Float, RawTerm);
impl_boxable!(BigInteger, RawTerm);
impl_boxable!(Reference, RawTerm);
impl_boxable!(ExternalPid, RawTerm);
impl_boxable!(ExternalPort, RawTerm);
impl_boxable!(ExternalReference, RawTerm);
impl_boxable!(Resource, RawTerm);
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
        (self.0 & !MASK_PRIMARY) as *const RawTerm as *mut RawTerm
    }
}

impl<T> Cast<*mut T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *mut T {
        assert!(self.is_boxed() || self.is_literal());
        (self.0 & !MASK_PRIMARY) as *const RawTerm as *mut T
    }
}

impl Cast<*mut Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *mut Cons {
        assert!(self.is_list());
        (self.0 & !MASK_PRIMARY) as *const RawTerm as *mut Cons
    }
}

impl Cast<*const RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *const RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
        (self.0 & !MASK_PRIMARY) as *const RawTerm
    }
}

impl<T> Cast<*const T> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> *const T {
        assert!(self.is_boxed() || self.is_literal());
        (self.0 & !MASK_PRIMARY) as *const T
    }
}

impl Cast<*const Cons> for RawTerm {
    #[inline]
    fn dyn_cast(self) -> *const Cons {
        assert!(self.is_list());
        (self.0 & !MASK_PRIMARY) as *const Cons
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
            Tag::Atom => Ok(TypedTerm::Atom(unsafe { self.decode_atom() })),
            Tag::Pid => Ok(TypedTerm::Pid(unsafe { self.decode_pid() })),
            Tag::Port => Ok(TypedTerm::Port(unsafe { self.decode_port() })),
            Tag::Box | Tag::Literal => {
                // NOTE: If the pointer we extract here is bogus or unmapped memory, we'll segfault,
                // but that is reflective of a bug where a term is being created or dereferenced incorrectly,
                // to find the source, you'll need to examine the trace to see where the input term is defined
                let ptr = (self.0 & !MASK_PRIMARY) as *const RawTerm;
                let unboxed = unsafe { *ptr };
                match unboxed.type_of() {
                    Tag::Nil => Ok(TypedTerm::Nil),
                    Tag::List => Ok(TypedTerm::List(unsafe { unboxed.decode_list() })),
                    Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe { unboxed.decode_smallint() })),
                    Tag::Atom => Ok(TypedTerm::Atom(unsafe { unboxed.decode_atom() })),
                    Tag::Pid => Ok(TypedTerm::Pid(unsafe { unboxed.decode_pid() })),
                    Tag::Port => Ok(TypedTerm::Port(unsafe { unboxed.decode_port() })),
                    Tag::Box | Tag::Literal => Err(TermDecodingError::MoveMarker.into()),
                    Tag::Unknown(_) => Err(TermDecodingError::InvalidTag.into()),
                    Tag::None => Err(TermDecodingError::NoneValue.into()),
                    header => unboxed.decode_header(header, Some(tag == Tag::Literal)),
                }
            }
            Tag::Unknown(_) => Err(TermDecodingError::InvalidTag.into()),
            Tag::None => Err(TermDecodingError::NoneValue.into()),
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
        self.0 & MASK_PRIMARY == FLAG_LITERAL
    }

    #[inline]
    fn is_list(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_LIST
    }

    #[inline]
    fn is_atom(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_ATOM
    }

    #[inline]
    fn is_smallint(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_SMALL_INTEGER
    }

    #[inline]
    fn is_bigint(self) -> bool {
        self.0 & MASK_HEADER == FLAG_BIG_INTEGER
    }

    #[inline]
    fn is_float(self) -> bool {
        self.0 & MASK_HEADER == FLAG_FLOAT
    }

    #[inline]
    fn is_function(self) -> bool {
        self.0 & MASK_HEADER == FLAG_CLOSURE
    }

    #[inline]
    fn is_tuple(self) -> bool {
        self.0 & MASK_HEADER == FLAG_TUPLE
    }

    #[inline]
    fn is_map(self) -> bool {
        self.0 & MASK_HEADER == FLAG_MAP
    }

    #[inline]
    fn is_local_pid(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_PID
    }

    #[inline]
    fn is_remote_pid(self) -> bool {
        self.0 & MASK_HEADER == FLAG_EXTERN_PID
    }

    #[inline]
    fn is_local_port(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_PORT
    }

    #[inline]
    fn is_remote_port(self) -> bool {
        self.0 & MASK_HEADER == FLAG_EXTERN_PORT
    }

    #[inline]
    fn is_local_reference(self) -> bool {
        self.0 & MASK_HEADER == FLAG_REFERENCE
    }

    #[inline]
    fn is_remote_reference(self) -> bool {
        self.0 & MASK_HEADER == FLAG_EXTERN_REF
    }

    #[inline]
    fn is_resource_reference(self) -> bool {
        self.0 & MASK_HEADER == FLAG_RESOURCE_REFERENCE
    }

    #[inline]
    fn is_procbin(self) -> bool {
        self.0 & MASK_HEADER == FLAG_PROCBIN
    }

    #[inline]
    fn is_heapbin(self) -> bool {
        self.0 & MASK_HEADER == FLAG_HEAPBIN
    }

    #[inline]
    fn is_subbinary(self) -> bool {
        self.0 & MASK_HEADER == FLAG_SUBBINARY
    }

    #[inline]
    fn is_match_context(self) -> bool {
        self.0 & MASK_HEADER == FLAG_MATCH_CTX
    }

    #[inline]
    fn is_boxed(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_BOXED
    }

    #[inline]
    fn is_header(self) -> bool {
        self.0 & MASK_PRIMARY == FLAG_HEADER
    }

    #[inline]
    fn is_immediate(self) -> bool {
        match self.type_of() {
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
                let ptr = (self.0 & !MASK_PRIMARY) as *const RawTerm;
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
            Tag::Box | Tag::Literal => {
                let ptr = (self.0 & !MASK_PRIMARY) as *const RawTerm;
                write!(f, "Box({:p})", ptr)
            }
            Tag::Unknown(invalid_tag) => {
                write!(f, "InvalidTerm(tag: {:064b})", invalid_tag)
            }
            header => {
                match self.decode_header(header, None) {
                    Ok(term) => write!(f, "Term({:?})", term),
                    Err(_) => write!(f, "InvalidHeader(tag: {:?})", header)
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
