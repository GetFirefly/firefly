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

const NUM_BITS: u64 = 64;

const TAG_SHIFT: u64 = 47;
pub const TAG_MASK: u64 = 0xFu64 << TAG_SHIFT;

const SUBTAG_SHIFT: u64 = TAG_SHIFT - 2;
const SUBTAG_MASK: u64 = 0xFCu64 << (TAG_SHIFT - 4);

const HEADER_SHIFT: u64 = 2;

// The highest allowed address in the pointer range
pub const MAX_ADDR: u64 = (1 << TAG_SHIFT) - 1;
// The beginning of the double range
pub const MIN_DOUBLE: u64 = !(i64::min_value() >> 12) as u64;
// The mask for the bits containing an immediate value
const IMMEDIATE_MASK: u64 = MAX_ADDR;
// The valid range of integer values that can fit in a term with primary + secondary tag
const MAX_HEADER_VALUE: u64 = MAX_ADDR >> HEADER_SHIFT;

pub struct Encoding64Nanboxed;

impl Encoding64Nanboxed {
    // Re-export this for use in ffi.rs
    pub const TAG_MASK: u64 = TAG_MASK;

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
    pub const TAG_LITERAL: u64 = 1;

    // We don't tag this type, they are implicit
    // const FLAG_FLOAT: u64 = u64::max_value();

    // Tags
    pub const TAG_BOXED: u64 = 0;
    pub const TAG_NONE: u64 = 0;
    pub const TAG_SMALL_INTEGER: u64 = 1 << TAG_SHIFT;
    pub const TAG_NIL: u64 = 2 << TAG_SHIFT;
    pub const TAG_LIST: u64 = 3 << TAG_SHIFT;
    pub const TAG_ATOM: u64 = 4 << TAG_SHIFT;
    pub const TAG_PID: u64 = 5 << TAG_SHIFT;
    pub const TAG_PORT: u64 = 6 << TAG_SHIFT;
    // Non-immediate tags
    pub const TAG_TUPLE: u64 = 7 << TAG_SHIFT;
    pub const TAG_BIG_INTEGER: u64 = 8 << TAG_SHIFT;
    pub const TAG_MAP: u64 = 9 << TAG_SHIFT;
    pub const TAG_REFERENCE: u64 = 10 << TAG_SHIFT;
    pub const TAG_CLOSURE: u64 = 11 << TAG_SHIFT;
    pub const TAG_RESOURCE_REFERENCE: u64 = 12 << TAG_SHIFT;

    pub const TAG_BINARY: u64 = 13 << TAG_SHIFT;
    pub const TAG_PROCBIN: u64 = Self::TAG_BINARY;
    pub const TAG_HEAPBIN: u64 = (1 << SUBTAG_SHIFT) | Self::TAG_BINARY;
    pub const TAG_SUBBINARY: u64 = (2 << SUBTAG_SHIFT) | Self::TAG_BINARY;
    pub const TAG_MATCH_CTX: u64 = (3 << SUBTAG_SHIFT) | Self::TAG_BINARY;

    pub const TAG_EXTERNAL: u64 = 14 << TAG_SHIFT;
    pub const TAG_EXTERN_PID: u64 = Self::TAG_EXTERNAL;
    pub const TAG_EXTERN_PORT: u64 = (1 << SUBTAG_SHIFT) | Self::TAG_EXTERNAL;
    pub const TAG_EXTERN_REF: u64 = (2 << SUBTAG_SHIFT) | Self::TAG_EXTERNAL;

    #[inline]
    pub fn encode_float(value: f64) -> u64 {
        value.to_bits() + MIN_DOUBLE
    }

    #[inline]
    pub fn decode_float(value: u64) -> f64 {
        debug_assert!(value >= MIN_DOUBLE);
        f64::from_bits(value - MIN_DOUBLE)
    }
}

impl Encoding for Encoding64Nanboxed {
    type Type = u64;
    type SignedType = i64;

    const MAX_IMMEDIATE_VALUE: u64 = MAX_ADDR;
    const MAX_ATOM_ID: u64 = Self::MAX_IMMEDIATE_VALUE;
    const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> (NUM_BITS - TAG_SHIFT);
    const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> (NUM_BITS - TAG_SHIFT);

    const NONE: u64 = 0;
    const NIL: u64 = 0 | Self::TAG_NIL;

    const FALSE: u64 = 0;
    const TRUE: u64 = 1;

    #[inline]
    fn type_of(value: u64) -> Tag<u64> {
        if value >= MIN_DOUBLE {
            return Tag::Float;
        } else if value == 0 {
            return Tag::None;
        } else if value == Self::TAG_NIL {
            return Tag::Nil;
        } else if value <= MAX_ADDR {
            if value & Self::TAG_LITERAL == Self::TAG_LITERAL {
                return Tag::Literal;
            } else {
                return Tag::Box;
            }
        } else {
            // There are 16 available tag combinations, and we use them all for valid values
            match value & TAG_MASK {
                // Binary types have a subtag of 2 bits, all 4 combinations are used
                Self::TAG_BINARY => match value & SUBTAG_MASK {
                    Self::TAG_PROCBIN => Tag::ProcBin,
                    Self::TAG_HEAPBIN => Tag::HeapBinary,
                    Self::TAG_SUBBINARY => Tag::SubBinary,
                    Self::TAG_MATCH_CTX => Tag::MatchContext,
                    tag => Tag::Unknown(tag),
                },
                // External types have a subtag of 2 bits, but only 3 combinations are used
                Self::TAG_EXTERNAL => match value & SUBTAG_MASK {
                    Self::TAG_EXTERN_PID => Tag::ExternalPid,
                    Self::TAG_EXTERN_PORT => Tag::ExternalPort,
                    Self::TAG_EXTERN_REF => Tag::ExternalReference,
                    tag => Tag::Unknown(tag),
                },
                Self::TAG_SMALL_INTEGER => Tag::SmallInteger,
                Self::TAG_ATOM => Tag::Atom,
                Self::TAG_PID => Tag::Pid,
                Self::TAG_PORT => Tag::Port,
                Self::TAG_LIST => Tag::List,
                Self::TAG_TUPLE => Tag::Tuple,
                Self::TAG_CLOSURE => Tag::Closure,
                Self::TAG_BIG_INTEGER => Tag::BigInteger,
                Self::TAG_REFERENCE => Tag::Reference,
                Self::TAG_RESOURCE_REFERENCE => Tag::ResourceReference,
                Self::TAG_MAP => Tag::Map,
                Self::TAG_NONE if value == Self::NONE => Tag::None,
                tag => Tag::Unknown(tag),
            }
        }
    }

    #[inline]
    fn immediate_mask_info() -> MaskInfo {
        MaskInfo {
            shift: 0,
            mask: IMMEDIATE_MASK,
        }
    }

    #[inline]
    fn encode_immediate(value: u64, tag: u64) -> u64 {
        debug_assert!(tag <= TAG_MASK, "invalid primary tag: {}", tag);
        debug_assert!(tag > MAX_ADDR, "invalid primary tag: {}", tag);
        debug_assert!(value <= MAX_ADDR, "invalid immediate value: {:064b}", value);
        value | tag
    }

    #[inline]
    fn encode_immediate_with_tag(value: u64, tag: Tag<u64>) -> u64 {
        match tag {
            Tag::Atom => Self::encode_immediate(value, Self::TAG_ATOM),
            Tag::Pid => Self::encode_immediate(value, Self::TAG_PID),
            Tag::Port => Self::encode_immediate(value, Self::TAG_PORT),
            Tag::SmallInteger => Self::encode_immediate(value, Self::TAG_SMALL_INTEGER),
            Tag::Float => value,
            Tag::Nil => Self::NIL,
            Tag::None => Self::NONE,
            _ => panic!("called encode_immediate_with_tag using non-immediate tag"),
        }
    }

    #[inline]
    fn encode_header(value: u64, tag: u64) -> u64 {
        debug_assert!(tag <= SUBTAG_MASK, "invalid header tag: {}", tag);
        debug_assert!(tag > MAX_HEADER_VALUE, "invalid header tag: {}", tag);
        debug_assert!(value <= MAX_HEADER_VALUE, "invalid header value: {}", value);
        value | tag
    }

    #[inline]
    fn encode_header_with_tag(value: u64, tag: Tag<u64>) -> u64 {
        match tag {
            Tag::BigInteger => Self::encode_header(value, Self::TAG_BIG_INTEGER),
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
    fn encode_list<T: ?Sized>(ptr: *const T) -> u64 {
        let value = ptr as *const () as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        value | Self::TAG_LIST
    }

    #[inline]
    fn encode_box<T: ?Sized>(ptr: *const T) -> u64 {
        let value = ptr as *const () as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        value | Self::TAG_BOXED
    }

    #[inline]
    fn encode_literal<T: ?Sized>(ptr: *const T) -> u64 {
        let value = ptr as *const () as u64;
        debug_assert!(
            value <= MAX_ADDR,
            "cannot encode pointers using more than 48 bits of addressable memory"
        );
        value | Self::TAG_LITERAL | Self::TAG_BOXED
    }

    #[inline]
    unsafe fn decode_box<T>(value: u64) -> *mut T {
        (value & !(TAG_MASK | Self::TAG_LITERAL)) as *const T as *mut T
    }

    #[inline]
    unsafe fn decode_list<T>(value: u64) -> *mut T {
        debug_assert_eq!(value & TAG_MASK, Self::TAG_LIST);
        (value & !TAG_MASK) as *const T as *mut T
    }

    #[inline]
    fn decode_smallint(value: u64) -> i64 {
        const SMALL_INTEGER_SIGNED: u64 = 1u64 << (TAG_SHIFT - 1);

        let value = value & !TAG_MASK;
        if value & SMALL_INTEGER_SIGNED == SMALL_INTEGER_SIGNED {
            (!MAX_ADDR | value) as i64
        } else {
            (MAX_ADDR & value) as i64
        }
    }

    #[inline]
    fn decode_immediate(value: u64) -> u64 {
        value & IMMEDIATE_MASK
    }

    #[inline]
    fn decode_header_value(value: u64) -> u64 {
        debug_assert!(
            value < MIN_DOUBLE,
            "invalid use of immediate float value as header"
        );
        debug_assert!(value > MAX_ADDR, "invalid use of boxed value as header");
        value & MAX_HEADER_VALUE
    }

    #[inline(always)]
    fn is_none(value: u64) -> bool {
        value == Self::NONE
    }

    #[inline(always)]
    fn is_nil(value: u64) -> bool {
        value == Self::TAG_NIL
    }

    #[inline]
    fn is_literal(value: u64) -> bool {
        value <= MAX_ADDR
            && value & Self::TAG_LITERAL == Self::TAG_LITERAL
            && (value & !Self::TAG_LITERAL > 0)
    }

    #[inline]
    fn is_list(value: u64) -> bool {
        !Self::is_float(value) && ((value == Self::TAG_NIL) || (value & TAG_MASK == Self::TAG_LIST))
    }

    #[inline]
    fn is_atom(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_ATOM)
    }

    #[inline]
    fn is_smallint(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_SMALL_INTEGER)
    }

    #[inline]
    fn is_bigint(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_BIG_INTEGER)
    }

    #[inline]
    fn is_float(value: u64) -> bool {
        value >= MIN_DOUBLE
    }

    #[inline]
    fn is_boxed_float(value: u64) -> bool {
        Self::is_float(value)
    }

    #[inline]
    fn is_function(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_CLOSURE)
    }

    #[inline]
    fn is_tuple(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_TUPLE)
    }

    #[inline]
    fn is_map(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_MAP)
    }

    #[inline]
    fn is_local_pid(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_PID)
    }

    #[inline]
    fn is_remote_pid(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_EXTERN_PID)
    }

    #[inline]
    fn is_local_port(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_PORT)
    }

    #[inline]
    fn is_remote_port(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_EXTERN_PORT)
    }

    #[inline]
    fn is_local_reference(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_REFERENCE)
    }

    #[inline]
    fn is_remote_reference(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_EXTERN_REF)
    }

    #[inline]
    fn is_resource_reference(value: u64) -> bool {
        !Self::is_float(value) && (value & TAG_MASK == Self::TAG_RESOURCE_REFERENCE)
    }

    #[inline]
    fn is_procbin(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_PROCBIN)
    }

    #[inline]
    fn is_heapbin(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_HEAPBIN)
    }

    #[inline]
    fn is_subbinary(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_SUBBINARY)
    }

    #[inline]
    fn is_match_context(value: u64) -> bool {
        !Self::is_float(value) && (value & SUBTAG_MASK == Self::TAG_MATCH_CTX)
    }

    #[inline]
    fn is_boxed(value: u64) -> bool {
        // >1 means it is not null or a null literal pointer
        value <= MAX_ADDR && value > 1
    }

    #[inline]
    fn is_header(value: u64) -> bool {
        // All headers have a tag + optional subtag, and all
        // header tags begin at FLAG_TUPLE and go up to the highest
        // tag value.
        !Self::is_float(value) && (value & SUBTAG_MASK) >= Self::TAG_TUPLE
    }

    #[inline]
    fn is_immediate(value: u64) -> bool {
        match Self::type_of(value) {
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
