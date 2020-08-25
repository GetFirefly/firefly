use crate::Tag;

use super::{Encoding, MaskInfo};

pub const PRIMARY_SHIFT: u32 = 3;
pub const HEADER_SHIFT: u32 = 8;
pub const HEADER_TAG_SHIFT: u32 = 3;

// The primary tag is given by masking bits 1-3
pub const MASK_PRIMARY: u32 = 0b111;
// Header is composed of 3 primary tag bits, and 4 subtag bits
pub const MASK_HEADER: u32 = 0b11111_111;
// The maximum allowed value to be stored in a header
pub const MAX_HEADER_VALUE: u32 = u32::max_value() >> HEADER_SHIFT;

pub struct Encoding32;

impl Encoding32 {
    // Re-export this for use in ffi.rs
    pub const MASK_PRIMARY: u32 = MASK_PRIMARY;

    // Primary tags (use lowest 3 bits, since minimum alignment is 8)
    pub const TAG_HEADER: u32 = 0; // 0b000
    pub const TAG_BOXED: u32 = 1; // 0b001
    pub const TAG_LIST: u32 = 2; // 0b010
    pub const TAG_LITERAL: u32 = 3; // 0b011
    pub const TAG_SMALL_INTEGER: u32 = 4; // 0b100
    pub const TAG_ATOM: u32 = 5; // 0b101
    pub const TAG_PID: u32 = 6; // 0b110
    pub const TAG_PORT: u32 = 7; // 0b111

    // Header tags (uses an additional 5 bits beyond the primary tag)
    // NONE is a special case where all bits of the header are zero
    pub const TAG_NONE: u32 = 0; // 0b00000_000
    pub const TAG_TUPLE: u32 = (1 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00001_000
    pub const TAG_BIG_INTEGER: u32 = (2 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00010_000
                                                                                 // const FLAG_UNUSED: u32 = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b00011_000
    pub const TAG_REFERENCE: u32 = (4 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00100_000
    pub const TAG_CLOSURE: u32 = (5 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00101_000
    pub const TAG_FLOAT: u32 = (6 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00110_000
    pub const TAG_RESOURCE_REFERENCE: u32 = (7 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b00111_000
    pub const TAG_PROCBIN: u32 = (8 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01000_000
    pub const TAG_HEAPBIN: u32 = (9 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01001_000
    pub const TAG_SUBBINARY: u32 = (10 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01010_000
    pub const TAG_MATCH_CTX: u32 = (11 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01011_000
    pub const TAG_EXTERN_PID: u32 = (12 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01100_000
    pub const TAG_EXTERN_PORT: u32 = (13 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01101_000
    pub const TAG_EXTERN_REF: u32 = (14 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01110_000
    pub const TAG_MAP: u32 = (15 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b01111_000
    pub const TAG_NIL: u32 = (16 << HEADER_TAG_SHIFT) | Self::TAG_HEADER; // 0b10000_000
}

impl Encoding for Encoding32 {
    type Type = u32;
    type SignedType = i32;

    const MAX_IMMEDIATE_VALUE: u32 = u32::max_value() >> 3;
    const MAX_ATOM_ID: u32 = Self::MAX_IMMEDIATE_VALUE;
    const MIN_SMALLINT_VALUE: i32 = i32::min_value() >> 4;
    const MAX_SMALLINT_VALUE: i32 = i32::max_value() >> 4;

    const NONE: u32 = 0;
    const NIL: u32 = 0 | Self::TAG_NIL;

    const FALSE: u32 = 0;
    const TRUE: u32 = 1;

    #[inline]
    fn type_of(value: u32) -> Tag<u32> {
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
            mask: MASK_PRIMARY as u64,
            max_allowed_value: Self::MAX_IMMEDIATE_VALUE as u64,
        }
    }

    #[inline]
    fn header_mask_info() -> MaskInfo {
        MaskInfo {
            shift: HEADER_SHIFT as i32,
            mask: 0,
            max_allowed_value: MAX_HEADER_VALUE as u64,
        }
    }

    #[inline]
    fn encode_immediate(value: u32, tag: u32) -> u32 {
        debug_assert!(tag <= MASK_PRIMARY, "invalid primary tag");
        (value << PRIMARY_SHIFT) | tag
    }

    #[inline]
    fn encode_immediate_with_tag(value: u32, tag: Tag<u32>) -> u32 {
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
    fn encode_list<T: ?Sized>(value: *const T) -> u32 {
        value as *const () as u32 | Self::TAG_LIST
    }

    #[inline]
    fn encode_box<T: ?Sized>(value: *const T) -> u32 {
        let ptr = value as *const () as u32;
        assert_eq!(ptr & MASK_PRIMARY, 0);
        ptr | Self::TAG_BOXED
    }

    #[inline]
    fn encode_literal<T: ?Sized>(value: *const T) -> u32 {
        let ptr = value as *const () as u32;
        assert_eq!(ptr & MASK_PRIMARY, 0);
        ptr | Self::TAG_LITERAL
    }

    #[inline]
    fn encode_header(value: u32, tag: u32) -> u32 {
        (value << HEADER_SHIFT) | tag
    }

    #[inline]
    fn encode_header_with_tag(value: u32, tag: Tag<u32>) -> u32 {
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
    unsafe fn decode_box<T>(value: u32) -> *mut T {
        (value & !MASK_PRIMARY) as *mut T
    }

    #[inline]
    unsafe fn decode_list<T>(value: u32) -> *mut T {
        debug_assert_eq!(value & MASK_PRIMARY, Self::TAG_LIST);
        (value & !MASK_PRIMARY) as *mut T
    }

    #[inline]
    fn decode_smallint(value: u32) -> i32 {
        let unmasked = (value & !MASK_PRIMARY) as i32;
        unmasked >> (PRIMARY_SHIFT as i32)
    }

    #[inline]
    fn decode_immediate(value: u32) -> u32 {
        (value & !MASK_PRIMARY) >> PRIMARY_SHIFT
    }

    #[inline]
    fn decode_header_value(value: u32) -> u32 {
        value >> HEADER_SHIFT
    }

    #[inline]
    fn is_none(value: u32) -> bool {
        if value == Self::NONE {
            return true;
        }
        // Check for null pointers
        match value & MASK_PRIMARY {
            Self::TAG_LITERAL | Self::TAG_BOXED | Self::TAG_LIST => {
                value & !MASK_PRIMARY == Self::NONE
            }
            _ => false,
        }
    }

    #[inline]
    fn is_nil(value: u32) -> bool {
        value == Self::TAG_NIL
    }

    #[inline]
    fn is_literal(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_LITERAL && value & !MASK_PRIMARY > 0
    }

    #[inline]
    fn is_list(value: u32) -> bool {
        value == Self::TAG_NIL || value & MASK_PRIMARY == Self::TAG_LIST
    }

    #[inline]
    fn is_atom(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_ATOM
    }

    #[inline]
    fn is_smallint(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_SMALL_INTEGER
    }

    #[inline]
    fn is_bigint(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_BIG_INTEGER
    }

    #[inline]
    fn is_float(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_FLOAT
    }

    #[inline]
    fn is_function(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_CLOSURE
    }

    #[inline]
    fn is_tuple(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_TUPLE
    }

    #[inline]
    fn is_map(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_MAP
    }

    #[inline]
    fn is_local_pid(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_PID
    }

    #[inline]
    fn is_remote_pid(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_PID
    }

    #[inline]
    fn is_local_port(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_PORT
    }

    #[inline]
    fn is_remote_port(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_PORT
    }

    #[inline]
    fn is_local_reference(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_REFERENCE
    }

    #[inline]
    fn is_remote_reference(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_EXTERN_REF
    }

    #[inline]
    fn is_resource_reference(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_RESOURCE_REFERENCE
    }

    #[inline]
    fn is_procbin(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_PROCBIN
    }

    #[inline]
    fn is_heapbin(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_HEAPBIN
    }

    #[inline]
    fn is_subbinary(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_SUBBINARY
    }

    #[inline]
    fn is_match_context(value: u32) -> bool {
        value & MASK_HEADER == Self::TAG_MATCH_CTX
    }

    #[inline]
    fn is_boxed(value: u32) -> bool {
        let tag = value & MASK_PRIMARY;
        (tag == Self::TAG_BOXED || tag == Self::TAG_LITERAL) && value & !MASK_PRIMARY != Self::NONE
    }

    #[inline]
    fn is_header(value: u32) -> bool {
        value & MASK_PRIMARY == Self::TAG_HEADER && value != Self::NONE && value != Self::NIL
    }

    #[inline]
    fn is_immediate(value: u32) -> bool {
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
