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
use crate::Tag;

use super::{Encoding, MaskInfo};

const PRIMARY_SHIFT: u64 = 3;
const HEADER_SHIFT: u64 = 8;
const HEADER_TAG_SHIFT: u64 = 3;

// The primary tag is given by masking bits 1-3
const MASK_PRIMARY: u64 = 0b111;
// Header is composed of 3 primary tag bits, and 4 subtag bits
const MASK_HEADER: u64 = 0b11111_111;

pub struct Encoding64;

impl Encoding64 {
    // Re-export this for use in ffi.rs
    pub const MASK_PRIMARY: u64 = MASK_PRIMARY;

    // Primary tags (use lowest 3 bits, since minimum alignment is 8)
    pub const TAG_HEADER: u64 = 0; // 0b000
    pub const TAG_BOXED: u64 = 1; // 0b001
    pub const TAG_LIST: u64 = 2; // 0b010
    pub const TAG_LITERAL: u64 = 3; // 0b011
    pub const TAG_SMALL_INTEGER: u64 = 4; // 0b100
    pub const TAG_ATOM: u64 = 5; // 0b101
    pub const TAG_PID: u64 = 6; // 0b110
    pub const TAG_PORT: u64 = 7; // 0b111

    // Header tags (uses an additional 5 bits beyond the primary tag)
    // NONE is a special case where all bits of the header are zero
    pub const TAG_NONE: u64 = 0; // 0b00000_000
    pub const TAG_TUPLE: u64 = (1 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00001_000
    pub const TAG_BIG_INTEGER: u64 = (2 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00010_000
    // const FLAG_UNUSED: u64 = (3 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00011_000
    pub const TAG_REFERENCE: u64 = (4 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00100_000
    pub const TAG_CLOSURE: u64 = (5 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00101_000
    pub const TAG_FLOAT: u64 = (6 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00110_000
    pub const TAG_RESOURCE_REFERENCE: u64 = (7 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00111_000
    pub const TAG_PROCBIN: u64 = (8 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01000_000
    pub const TAG_HEAPBIN: u64 = (9 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01001_000
    pub const TAG_SUBBINARY: u64 = (10 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01010_000
    pub const TAG_MATCH_CTX: u64 = (11 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01011_000
    pub const TAG_EXTERN_PID: u64 = (12 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01100_000
    pub const TAG_EXTERN_PORT: u64 = (13 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01101_000
    pub const TAG_EXTERN_REF: u64 = (14 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01110_000
    pub const TAG_MAP: u64 = (15 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01111_000
    pub const TAG_NIL: u64 = (16 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b10000_000
}

impl Encoding for Encoding64 {
    type Type = u64;
    type SignedType = i64;

    const MAX_IMMEDIATE_VALUE: u64 = u64::max_value() >> 3;
    const MAX_ATOM_ID: u64 = Self::MAX_IMMEDIATE_VALUE;
    const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> 4;
    const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> 4;

    const NONE: u64 = 0;
    const NIL: u64 = 0 | Self::TAG_NIL;

    const FALSE: u64 = 0;
    const TRUE: u64 = 1;

    #[inline]
    fn type_of(value: u64) -> Tag<u64> {
        let tag = match value & MASK_PRIMARY {
            Self::TAG_HEADER => value & MASK_HEADER,
            tag => tag,
        };

        match tag {
            Self::TAG_BOXED => {
                if value & !MASK_PRIMARY == Self::NONE {
                    Tag::None
                } else {
                    Tag::Box
                }
            }
            Self::TAG_LITERAL => {
                if value & !MASK_PRIMARY == Self::NONE {
                    Tag::None
                } else {
                    Tag::Literal
                }
            }
            Self::TAG_NIL => Tag::Nil,
            Self::TAG_SMALL_INTEGER => Tag::SmallInteger,
            Self::TAG_ATOM => Tag::Atom,
            Self::TAG_PID => Tag::Pid,
            Self::TAG_PORT => Tag::Port,
            Self::TAG_LIST => Tag::List,
            Self::TAG_TUPLE => Tag::Tuple,
            Self::TAG_CLOSURE => Tag::Closure,
            Self::TAG_HEAPBIN => Tag::HeapBinary,
            Self::TAG_BIG_INTEGER => Tag::BigInteger,
            Self::TAG_REFERENCE => Tag::Reference,
            Self::TAG_FLOAT => Tag::Float,
            Self::TAG_RESOURCE_REFERENCE => Tag::ResourceReference,
            Self::TAG_PROCBIN => Tag::ProcBin,
            Self::TAG_SUBBINARY => Tag::SubBinary,
            Self::TAG_MATCH_CTX => Tag::MatchContext,
            Self::TAG_EXTERN_PID => Tag::ExternalPid,
            Self::TAG_EXTERN_PORT => Tag::ExternalPort,
            Self::TAG_EXTERN_REF => Tag::ExternalReference,
            Self::TAG_MAP => Tag::Map,
            Self::TAG_NONE if value == Self::NONE => Tag::None,
            _ => Tag::Unknown(tag),
        }
    }

    #[inline]
    fn immediate_mask_info() -> MaskInfo {
        MaskInfo {
            shift: PRIMARY_SHIFT as i32,
            mask: MASK_PRIMARY,
        }
    }

    #[inline]
    fn encode_immediate(value: u64, tag: u64) -> u64 {
        debug_assert!(tag <= MASK_PRIMARY, "invalid primary tag");
        (value << PRIMARY_SHIFT) | tag
    }

    #[inline]
    fn encode_immediate_with_tag(value: u64, tag: Tag<u64>) -> u64 {
        match tag {
            Tag::Atom => Self::encode_immediate(value, Self::TAG_ATOM),
            Tag::Pid => Self::encode_immediate(value, Self::TAG_PID),
            Tag::Port => Self::encode_immediate(value, Self::TAG_PORT),
            Tag::SmallInteger => Self::encode_immediate(value, Self::TAG_SMALL_INTEGER),
            Tag::Nil => Self::NIL,
            Tag::None => Self::NONE,
            _ => panic!("called encode_immediate_with_tag using non-immediate tag"),
        }
    }

    #[inline]
    fn encode_list<T: ?Sized>(value: *const T) -> u64 {
        value as *const () as u64 | Self::TAG_LIST
    }

    #[inline]
    fn encode_box<T: ?Sized>(value: *const T) -> u64 {
        let ptr = value as *const () as u64;
        assert_eq!(ptr & MASK_PRIMARY, 0);
        ptr | Self::TAG_BOXED
    }

    #[inline]
    fn encode_literal<T: ?Sized>(value: *const T) -> u64 {
        let ptr = value as *const () as u64;
        assert_eq!(ptr & MASK_PRIMARY, 0);
        ptr | Self::TAG_LITERAL
    }

    #[inline]
    fn encode_header(value: u64, tag: u64) -> u64 {
        (value << HEADER_SHIFT) | tag
    }

    #[inline]
    fn encode_header_with_tag(value: u64, tag: Tag<u64>) -> u64 {
        match tag {
            Tag::BigInteger => Self::encode_header(value, Self::TAG_BIG_INTEGER),
            Tag::Float => Self::encode_header(value, Self::TAG_FLOAT),
            Tag::Tuple => Self::encode_header(value, Self::TAG_TUPLE),
            Tag::Map => Self::encode_header(value, Self::TAG_MAP),
            Tag::Closure => Self::encode_header(value, Self::TAG_CLOSURE),
            Tag::ProcBin => Self::encode_header(value, Self::TAG_PROCBIN),
            Tag::HeapBinary => Self::encode_header(value, Self::TAG_HEAPBIN),
            Tag::SubBinary => Self::encode_header(value, Self::TAG_SUBBINARY),
            Tag::MatchContext => Self::encode_header(value, Self::TAG_MATCH_CTX),
            Tag::ExternalPid => Self::encode_header(value, Self::TAG_EXTERN_PID),
            Tag::ExternalPort => Self::encode_header(value, Self::TAG_EXTERN_PORT),
            Tag::ExternalReference => Self::encode_header(value, Self::TAG_EXTERN_REF),
            Tag::Reference => Self::encode_header(value, Self::TAG_REFERENCE),
            Tag::ResourceReference => Self::encode_header(value, Self::TAG_RESOURCE_REFERENCE),
            _ => panic!("called encode_header_with_tag using non-boxable tag"),
        }
    }

    #[inline]
    unsafe fn decode_box<T>(value: u64) -> *mut T {
        (value & !MASK_PRIMARY) as *mut T
    }

    #[inline]
    unsafe fn decode_list<T>(value: u64) -> *mut T {
        debug_assert_eq!(value & MASK_PRIMARY, Self::TAG_LIST);
        (value & !MASK_PRIMARY) as *mut T
    }

    #[inline]
    fn decode_smallint(value: u64) -> i64 {
        let unmasked = (value & !MASK_PRIMARY) as i64;
        unmasked >> 3
    }

    #[inline]
    fn decode_immediate(value: u64) -> u64 {
        (value & !MASK_PRIMARY) >> PRIMARY_SHIFT
    }

    #[inline]
    fn decode_header_value(value: u64) -> u64 {
        value >> HEADER_SHIFT
    }

    #[inline]
    fn is_none(value: u64) -> bool {
        if value == Self::NONE {
            return true;
        }
        // Check for null pointers
        match value & MASK_PRIMARY {
            Self::TAG_LITERAL | Self::TAG_BOXED | Self::TAG_LIST => value & !MASK_PRIMARY == Self::NONE,
            _ => false,
        }
    }

    #[inline]
    fn is_nil(value: u64) -> bool {
        value == Self::TAG_NIL
    }

    #[inline]
    fn is_literal(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_LITERAL && value & !MASK_PRIMARY > 0
    }

    #[inline]
    fn is_list(value: u64) -> bool {
        value == Self::TAG_NIL || value & MASK_PRIMARY == Self::TAG_LIST
    }

    #[inline]
    fn is_atom(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_ATOM
    }

    #[inline]
    fn is_smallint(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_SMALL_INTEGER
    }

    #[inline]
    fn is_bigint(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_BIG_INTEGER
    }

    #[inline]
    fn is_float(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_FLOAT
    }

    #[inline]
    fn is_function(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_CLOSURE
    }

    #[inline]
    fn is_tuple(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_TUPLE
    }

    #[inline]
    fn is_map(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_MAP
    }

    #[inline]
    fn is_local_pid(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_PID
    }

    #[inline]
    fn is_remote_pid(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_PID
    }

    #[inline]
    fn is_local_port(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_PORT
    }

    #[inline]
    fn is_remote_port(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_PORT
    }

    #[inline]
    fn is_local_reference(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_REFERENCE
    }

    #[inline]
    fn is_remote_reference(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_REF
    }

    #[inline]
    fn is_resource_reference(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_RESOURCE_REFERENCE
    }

    #[inline]
    fn is_procbin(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_PROCBIN
    }

    #[inline]
    fn is_heapbin(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_HEAPBIN
    }

    #[inline]
    fn is_subbinary(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_SUBBINARY
    }

    #[inline]
    fn is_match_context(value: u64) -> bool {
        value & MASK_HEADER == Self::TAG_MATCH_CTX
    }

    #[inline]
    fn is_boxed(value: u64) -> bool {
        let tag = value & MASK_PRIMARY;
        (tag == Self::TAG_BOXED || tag == Self::TAG_LITERAL) && value & !MASK_PRIMARY != Self::NONE
    }

    #[inline]
    fn is_header(value: u64) -> bool {
        value & MASK_PRIMARY == Self::TAG_HEADER && value != Self::NONE
    }

    #[inline]
    fn is_immediate(value: u64) -> bool {
        match Self::type_of(value) {
            Tag::SmallInteger => true,
            Tag::Atom => true,
            Tag::Pid => true,
            Tag::Port => true,
            Tag::Nil => true,
            _ => false,
        }
    }
}
