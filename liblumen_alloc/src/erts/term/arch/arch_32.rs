use core::cmp;
use core::convert::TryInto;
///! This module exposes 32-bit architecture specific values and functions
///!
///! See the module doc in arch_64.rs for more information
use core::fmt;

use crate::erts::exception::{Exception, Result};

use liblumen_core::sys::sysconf::MIN_ALIGN;
const_assert!(MIN_ALIGN >= 4);

use crate::erts::term::prelude::*;

use super::{Repr, Tag};

#[cfg_attr(not(target_pointer_width = "32"), allow(unused))]
pub type Word = u32;

// The valid range of integer values that can fit in a term with primary tag
#[cfg_attr(not(target_pointer_width = "32"), allow(unused))]
pub const MAX_IMMEDIATE_VALUE: u32 = u32::max_value() >> 3;
#[cfg_attr(not(target_pointer_width = "32"), allow(unused))]
pub const MAX_ATOM_ID: u32 = MAX_IMMEDIATE_VALUE;

// The valid range of fixed-width integers
#[cfg_attr(not(target_pointer_width = "32"), allow(unused))]
pub const MIN_SMALLINT_VALUE: i32 = i32::min_value() >> 4;
#[cfg_attr(not(target_pointer_width = "32"), allow(unused))]
pub const MAX_SMALLINT_VALUE: i32 = i32::max_value() >> 4;

const PRIMARY_SHIFT: u32 = 3;
const HEADER_SHIFT: u32 = 8;
const HEADER_TAG_SHIFT: u32 = 3;

// Primary tags (use lowest 3 bits, since minimum alignment is 8)
const FLAG_HEADER: u32 = 0; // 0b000
const FLAG_BOXED: u32 = 1; // 0b001
const FLAG_LIST: u32 = 2; // 0b010
const FLAG_LITERAL: u32 = 3; // 0b011
const FLAG_SMALL_INTEGER: u32 = 4; // 0b100
const FLAG_ATOM: u32 = 5; // 0b101
const FLAG_PID: u32 = 6; // 0b110
const FLAG_PORT: u32 = 7; // 0b111

// Header tags (uses an additional 5 bits beyond the primary tag)
// NONE is a special case where all bits of the header are zero
const FLAG_NONE: u32 = 0; // 0b00000_000
const FLAG_TUPLE: u32 = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00001_000
const FLAG_BIG_INTEGER: u32 = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00010_000
#[allow(unused)]
const FLAG_UNUSED: u32 = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00011_000
const FLAG_REFERENCE: u32 = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00100_000
const FLAG_CLOSURE: u32 = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00101_000
const FLAG_FLOAT: u32 = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00110_000
const FLAG_RESOURCE_REFERENCE: u32 = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00111_000
const FLAG_PROCBIN: u32 = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01000_000
const FLAG_HEAPBIN: u32 = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01001_000
const FLAG_SUBBINARY: u32 = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01010_000
const FLAG_MATCH_CTX: u32 = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01011_000
const FLAG_EXTERN_PID: u32 = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01100_000
const FLAG_EXTERN_PORT: u32 = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01101_000
const FLAG_EXTERN_REF: u32 = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01110_000
const FLAG_MAP: u32 = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b01111_000
const FLAG_NIL: u32 = (16 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b10000_000

// The primary tag is given by masking bits 1-3
const MASK_PRIMARY: u32 = 0b111;
// Header is composed of 3 primary tag bits, and 4 subtag bits
const MASK_HEADER: u32 = 0b11111_111;

const NONE: u32 = 0;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct RawTerm(u32);
impl RawTerm {
    pub const NONE: Self = Self(NONE);
    pub const NIL: Self = Self(FLAG_NIL);

    pub const HEADER_TUPLE: u32 = FLAG_TUPLE;
    pub const HEADER_BIG_INTEGER: u32 = FLAG_BIG_INTEGER;
    pub const HEADER_REFERENCE: u32 = FLAG_REFERENCE;
    pub const HEADER_CLOSURE: u32 = FLAG_CLOSURE;
    pub const HEADER_FLOAT: u32 = FLAG_FLOAT;
    pub const HEADER_RESOURCE_REFERENCE: u32 = FLAG_RESOURCE_REFERENCE;
    pub const HEADER_PROCBIN: u32 = FLAG_PROCBIN;
    pub const HEADER_BINARY_LITERAL: u32 = FLAG_PROCBIN;
    pub const HEADER_HEAPBIN: u32 = FLAG_HEAPBIN;
    pub const HEADER_SUBBINARY: u32 = FLAG_SUBBINARY;
    pub const HEADER_MATCH_CTX: u32 = FLAG_MATCH_CTX;
    pub const HEADER_EXTERN_PID: u32 = FLAG_EXTERN_PID;
    pub const HEADER_EXTERN_PORT: u32 = FLAG_EXTERN_PORT;
    pub const HEADER_EXTERN_REF: u32 = FLAG_EXTERN_REF;
    pub const HEADER_MAP: u32 = FLAG_MAP;
}
impl fmt::Binary for RawTerm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#b}", self.value())
    }
}
impl Repr for RawTerm {
    type Word = u32;

    #[inline]
    fn as_usize(&self) -> usize {
        self.0 as usize
    }

    #[inline]
    fn word_to_usize(word: u32) -> usize {
        word as usize
    }

    #[inline]
    fn value(&self) -> u32 {
        self.0
    }

    #[inline]
    fn type_of(&self) -> Tag<u32> {
        let term = self.0;
        let tag = match term & MASK_PRIMARY {
            FLAG_HEADER => term & MASK_HEADER,
            tag => tag,
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
            _ => Tag::Unknown(tag),
        }
    }

    #[inline]
    fn encode_immediate(value: u32, tag: u32) -> Self {
        debug_assert!(tag <= MASK_PRIMARY, "invalid primary tag");
        Self((value << PRIMARY_SHIFT) | tag)
    }

    #[inline]
    fn encode_list(value: *const Cons) -> Self {
        Self(value as u32 | FLAG_LIST)
    }

    #[inline]
    fn encode_box<T>(value: *const T) -> Self
    where
        T: ?Sized,
    {
        Self(value as *const () as u32 | FLAG_BOXED)
    }

    #[inline]
    fn encode_literal<T>(value: *const T) -> Self
    where
        T: ?Sized,
    {
        Self(value as *const () as u32 | FLAG_LITERAL)
    }

    #[inline]
    fn encode_header(value: u32, tag: u32) -> Self {
        Self((value << HEADER_SHIFT) | tag)
    }

    #[inline]
    unsafe fn decode_box(self) -> *mut Self {
        (self.0 & !MASK_PRIMARY) as *const RawTerm as *mut RawTerm
    }

    #[inline]
    unsafe fn decode_list(self) -> Boxed<Cons> {
        debug_assert_eq!(self.0 & MASK_PRIMARY, FLAG_LIST);
        let ptr = (self.0 & !MASK_PRIMARY) as *mut Cons;
        Boxed::new_unchecked(ptr)
    }

    #[inline]
    unsafe fn decode_smallint(self) -> SmallInteger {
        let unmasked = (self.0 & !MASK_PRIMARY) as i32;
        let i = unmasked >> 3;
        SmallInteger::new_unchecked(i as isize)
    }

    #[inline]
    unsafe fn decode_immediate(self) -> u32 {
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
    unsafe fn decode_header_value(&self) -> u32 {
        debug_assert_eq!(self.0 & MASK_HEADER, 0);
        self.0 >> HEADER_SHIFT
    }
}

unsafe impl Send for RawTerm {}

impl Encode<RawTerm> for u8 {
    fn encode(&self) -> Result<RawTerm> {
        Ok(RawTerm::encode_immediate(
            (*self) as u32,
            FLAG_SMALL_INTEGER,
        ))
    }
}

impl Encode<RawTerm> for SmallInteger {
    fn encode(&self) -> Result<RawTerm> {
        let i: i32 = (*self)
            .try_into()
            .map_err(|_| Exception::from(TermEncodingError::ValueOutOfRange))?;
        Ok(RawTerm::encode_immediate(i as u32, FLAG_SMALL_INTEGER))
    }
}

impl Encode<RawTerm> for bool {
    fn encode(&self) -> Result<RawTerm> {
        let atom = Atom::try_from_str(&self.to_string()).unwrap();
        Ok(RawTerm::encode_immediate(atom.id() as u32, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Atom {
    fn encode(&self) -> Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.id() as u32, FLAG_ATOM))
    }
}

impl Encode<RawTerm> for Pid {
    fn encode(&self) -> Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.as_usize() as u32, FLAG_PID))
    }
}

impl Encode<RawTerm> for Port {
    fn encode(&self) -> Result<RawTerm> {
        Ok(RawTerm::encode_immediate(self.as_usize() as u32, FLAG_PORT))
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
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
        unsafe { self.decode_box() }
    }
}

impl<T> Cast<Boxed<T>> for RawTerm
where
    T: Boxable<RawTerm>,
{
    #[inline]
    default fn dyn_cast(self) -> Boxed<T> {
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
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
        assert!(self.is_list());
        unsafe { self.decode_box() as *mut Cons }
    }
}

impl Cast<*const RawTerm> for RawTerm {
    #[inline]
    default fn dyn_cast(self) -> *const RawTerm {
        assert!(self.is_boxed() || self.is_literal() || self.is_list());
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
        assert!(self.is_list());
        unsafe { self.decode_box() as *const Cons }
    }
}

impl Encoded for RawTerm {
    #[inline]
    fn decode(&self) -> Result<TypedTerm> {
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
                // but that is reflective of a bug where a term is being created or dereferenced
                // incorrectly, to find the source, you'll need to examine the trace
                // to see where the input term is defined
                let ptr = unsafe { self.decode_box() };
                let unboxed = unsafe { *ptr };
                match unboxed.type_of() {
                    Tag::Nil => Ok(TypedTerm::Nil),
                    Tag::List => Ok(TypedTerm::List(unsafe { unboxed.decode_list() })),
                    Tag::SmallInteger => Ok(TypedTerm::SmallInteger(unsafe {
                        unboxed.decode_smallint()
                    })),
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
            Tag::Unknown(invalid_tag) => write!(f, "InvalidTerm(tag: {:064b})", invalid_tag),
            header => match self.decode_header(header, None) {
                Ok(term) => write!(f, "Term({:?})", term),
                Err(_) => write!(f, "InvalidHeader(tag: {:?})", header),
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

#[cfg(all(test, target_pointer_width = "32"))]
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

        assert!(literal_boxed.is_boxed());
        assert!(literal_boxed.is_literal());
        assert_eq!(literal_boxed.type_of(), Tag::Literal);
        assert!(!literal_boxed.is_header());
        assert!(!literal_boxed.is_immediate());
    }

    #[test]
    fn float_encoding_x86_64() {
        let float: Float = std::f64::MAX.into();

        let float = heap.float(float).unwrap();
        let float_term: RawTerm = float.encode().unwrap();
        assert!(float_term.is_boxed());
        assert_eq!(float_term.type_of(), Tag::Box);
        assert!(!float_term.is_float());

        let unboxed: *const RawTerm = float_term.dyn_cast();
        let float_header = unsafe { *unboxed };
        assert!(float_header.is_header());
        assert!(float_header.is_float());
        assert_eq!(float_header.type_of(), Tag::Float);

        let float_decoded: Result<Boxed<Float>, _> = float_term.decode().unwrap().try_into();
        assert!(float_decoded.is_ok());
        let float_box = float_decoded.unwrap();
        assert_eq!(&float, float_box.as_ref());
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
        let pid = unsafe { Pid::from_raw(MAX_IMMEDIATE_VALUE as usize) };

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
        let port = unsafe { Port::from_raw(IMMEDIATE_VALUE_ADDR as usize) };

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
        assert!(!RawTerm::NIL.is_list());
        assert_eq!(RawTerm::NIL.type_of(), Tag::Nil);
        assert!(RawTerm::NIL.is_nil());
        assert!(RawTerm::NIL.is_immediate());

        // Non-empty list
        let list = cons!(heap, fixnum!(1), fixnum!(2));
        let list_term: RawTerm = list.encode().unwrap();
        assert!(!list_term.is_boxed());
        assert!(list_term.is_list());
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
        use crate::erts::ModuleFunctionArity;
        use alloc::sync::Arc;

        let mut heap = RegionHeap::default();
        let creator = Pid::make_term(0, 0).unwrap();

        let module = Atom::try_from_str("module").unwrap();
        let function = Atom::try_from_str("function").unwrap();
        let arity = 0;
        let mfa = Arc::new(ModuleFunctionArity {
            module: module.clone(),
            function,
            arity,
        });
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
                2,
                Some(code),
                Creator::Local(creator),
                &[&[one, two]],
            )
            .unwrap();
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
        assert_eq!(closure_box.arity(), 0);
        assert_eq!(closure_box.module_function_arity(), mfa);
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
        let mut heap = RegionHeap::default();

        let pid = ExternalPid::with_node_id(1, 2, 3).unwrap();
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
