#![allow(unused)]
///! This module contains constants for 64-bit architectures used by the term
///! implementation.
///! On 32-bit platforms we generally can use the high bits like we
///! do on 64-bit platforms, as the full address space is rarely available,
///! e.g. Windows with 2-3gb addressable range, Linux with 3gb. But to avoid
///! the reliance on that fact, we use the low bits on 32-bit platforms
///! for the pointer-typed terms. We have 3 bits available to us because we
///! require an 8-byte minimum alignment for all allocations, ensuring that
///! the lowest 3 bits are always zeroes, and thus available for tags. For
///! non-pointer terms, the flags all go in the high bits, to make accessing
///! the value and tags as easy as applying a mask, no shifts needed.
use core::mem;
use crate::erts::term::Term;
use crate::erts::term::list::Cons;

pub const NUM_BITS: u32 = 32;
pub const MIN_ALIGNMENT: u32 = 8;

const MAX_LOGICAL_ALIGNED_ADDR: u32 = u32::max_value() & !(MIN_ALIGNMENT - 1);
/// This is the highest assignable 8-byte aligned address on this architecture
///
/// NOTE: On Windows and Linux, this address is actually capped at 2gb and 3gb respectively,
/// but other platforms may not have this restriction, so our tagging scheme tries to avoid
/// using high bits if at all possible. We do use the highest bit as a flag for literal values,
/// i.e. pointers to literal constants
pub const MAX_ALIGNED_ADDR: u32 = MAX_LOGICAL_ALIGNED_ADDR - (1u32 << (NUM_BITS - 1));

const IMMEDIATE1_SHIFT: u32 = 3;
const IMMEDIATE1_VALUE_SHIFT: u32 = 5;
const IMMEDIATE2_SHIFT: u32 = 5;
const IMMEDIATE2_VALUE_SHIFT: u32 = 7;
const HEADER_TAG_SHIFT: u32 = 2;
const HEADER_VALUE_SHIFT: u32 = 6;

// This is the largest value that will fit in a first-class immediate (i.e. pid)
// Integer immediates are handled specially, due to the extra bit required for signing
pub const MAX_IMMEDIATE1_VALUE: u32 = u32::max_value() >> (NUM_BITS - (NUM_BITS - IMMEDIATE1_VALUE_SHIFT));
// This is the largest value that will fit in a second-class immediate (i.e. atom)
pub const MAX_IMMEDIATE2_VALUE: u32 = u32::max_value() >> (NUM_BITS - (NUM_BITS - IMMEDIATE2_VALUE_SHIFT));
// Small integers require an extra bit for the sign, 6 bits, i.e. they are effectively 26bit
pub const MIN_SMALLINT_VALUE: i32 = i32::min_value() >> (NUM_BITS - (NUM_BITS - 6));
pub const MAX_SMALLINT_VALUE: i32 = i32::max_value() >> (NUM_BITS - (NUM_BITS - 6));
// Atom IDs are unsigned integer values and stored in a second-class immediate
pub const MAX_ATOM_ID: u32 = MAX_IMMEDIATE2_VALUE;

// Primary types
pub const FLAG_HEADER: u32 = 0;
pub const FLAG_LIST: u32 = 1 << 0; // 0b001
pub const FLAG_BOXED: u32 = 1 << 1; // 0b010
pub const FLAG_LITERAL: u32 = 1 << 2; // 0b100
pub const FLAG_IMMEDIATE: u32 = 3 << 0; // 0b011
pub const FLAG_IMMEDIATE2: u32 = (2 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b10_011
                                                                                // First class immediates
pub const FLAG_PID: u32 = 0 | FLAG_IMMEDIATE; // 0b00_011
pub const FLAG_PORT: u32 = (1 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b01_011
pub const FLAG_SMALL_INTEGER: u32 = (3 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b11_011
pub const FLAG_SMALL_INTEGER_SIGN: u32 = (1 << IMMEDIATE1_VALUE_SHIFT) | FLAG_SMALL_INTEGER;
// Second class immediates
pub const FLAG_ATOM: u32 = 0 | FLAG_IMMEDIATE2; // 0b0010_011
pub const FLAG_CATCH: u32 = (1 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b0110_011
pub const FLAG_UNUSED_1: u32 = (2 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b1010_011
pub const FLAG_NIL: u32 = (3 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b1110_011
                                                                        // Header types, these flags re-use the literal flag bit, as it only applies to primary tags
pub const FLAG_TUPLE: u32 = 0 | FLAG_HEADER; // 0b000_000
pub const FLAG_NONE: u32 = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b000_100
pub const FLAG_POS_BIG_INTEGER: u32 = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b001_000
pub const FLAG_NEG_BIG_INTEGER: u32 = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b001_100
pub const FLAG_REFERENCE: u32 = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b010_000
pub const FLAG_CLOSURE: u32 = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b010_100
pub const FLAG_FLOAT: u32 = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b011_000
pub const FLAG_RESOURCE_REFERENCE: u32 = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b011_100
pub const FLAG_PROCBIN: u32 = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b100_000
pub const FLAG_HEAPBIN: u32 = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b100_100
pub const FLAG_SUBBINARY: u32 = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b101_000
pub const FLAG_MATCH_CTX: u32 = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b101_100
pub const FLAG_EXTERN_PID: u32 = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b110_000
pub const FLAG_EXTERN_PORT: u32 = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b110_100
pub const FLAG_EXTERN_REF: u32 = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b111_000
pub const FLAG_MAP: u32 = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b111_100

// The primary tag is given by masking bits 1-2
pub const MASK_PRIMARY: u32 = 0x3;
// The literal flag is at bit 3
pub const MASK_LITERAL: u32 = 0x4;
// First class immediate tags are given by masking bits 4-5
pub const MASK_IMMEDIATE1_TAG: u32 = 0x18;
// Second class immediate tags are given by masking bits 6-7
pub const MASK_IMMEDIATE2_TAG: u32 = 0x60;
// To mask off the entire immediate header, we mask off the primary, listeral, and immediate tag
pub const MASK_IMMEDIATE1: u32 = MASK_PRIMARY | MASK_LITERAL | MASK_IMMEDIATE1_TAG;
pub const MASK_IMMEDIATE2: u32 = MASK_IMMEDIATE1 | MASK_IMMEDIATE2_TAG;
// Header is composed of 2 primary tag bits, and 4 subtag bits, re-using the literal flag bit
pub const MASK_HEADER: u32 = 0x3F;
// The arityval is a subtag that identifies the boxed type
// This value is used as a marker in some checks, but it is essentially equivalent
// to `FLAG_TUPLE & !FLAG_HEADER`, which is simply the value 0
pub const ARITYVAL: u32 = 0;
// The following is a mask for the arityval subtag value
pub const MASK_HEADER_ARITYVAL: u32 = 0x3C;

/// The pattern 0b0101 out to usize bits, but with the header bits
/// masked out, and flagged as the none value
pub const NONE_VAL: u32 = 0x55555555u32 & !MASK_HEADER;

#[inline]
pub const fn is_literal(term: u32) -> bool {
    term & FLAG_LITERAL == FLAG_LITERAL
}

#[inline]
pub const fn primary_tag(term: u32) -> u32 {
    term & MASK_PRIMARY
}

#[inline]
pub const fn make_header(term: u32, tag: u32) -> u32 {
    (term << HEADER_VALUE_SHIFT) | tag
}

#[inline]
pub const fn header_tag(term: u32) -> u32 {
    term & MASK_HEADER
}

#[inline]
pub const fn header_arityval_tag(term: u32) -> u32 {
    term & MASK_HEADER_ARITYVAL
}

#[inline]
pub const fn header_value(term: u32) -> u32 {
    (term & !MASK_HEADER) >> HEADER_VALUE_SHIFT
}

#[inline]
pub const fn make_immediate1(value: u32, tag: u32) -> u32 {
    (value << IMMEDIATE1_VALUE_SHIFT) | tag
}

#[inline]
pub const fn make_immediate2(value: u32, tag: u32) -> u32 {
    (value << IMMEDIATE2_VALUE_SHIFT) | tag
}

#[inline]
pub const fn immediate1_tag(term: u32) -> u32 {
    term & MASK_IMMEDIATE1
}

#[inline]
pub const fn immediate2_tag(term: u32) -> u32 {
    term & MASK_IMMEDIATE2
}

#[inline]
pub const fn immediate1_value(term: u32) -> u32 {
    (term & !MASK_IMMEDIATE1) >> IMMEDIATE1_VALUE_SHIFT
}

#[inline]
pub const fn immediate2_value(term: u32) -> u32 {
    (term & !MASK_IMMEDIATE2) >> IMMEDIATE1_VALUE_SHIFT
}

#[inline]
pub fn make_smallint(value: i32) -> u32 {
    debug_assert!(value <= MAX_SMALLINT_VALUE);
    debug_assert!(value >= MIN_SMALLINT_VALUE);
    match value.signum() {
        0 | 1 => make_immediate1(value as u32, FLAG_SMALL_INTEGER),
        _ => make_immediate1(value as u32, FLAG_SMALL_INTEGER | FLAG_SMALL_INTEGER_SIGN)
    }
}

#[inline]
pub fn smallint_value(term: u32) -> i32 {
    if term & FLAG_SMALL_INTEGER_SIGN == FLAG_SMALL_INTEGER_SIGN {
        // Signed, i.e. negative
        // Unlike 64-bit, negative numbers require shifting and
        // sign extending to get back the original number
        // First, strip tags and unshift
        let needs_signext = ((term & !(FLAG_SMALL_INTEGER | FLAG_SMALL_INTEGER_SIGN)) >> IMMEDIATE1_VALUE_SHIFT);
        // Then sign-extend to restore the original value
        (needs_signext | (0x1F << (NUM_BITS - IMMEDIATE1_VALUE_SHIFT))) as i32
    } else {
        // Unsigned, i.e. positive or zero
        // This is almost the same as the negative case, except we don't sign extend
        ((term & !FLAG_SMALL_INTEGER) >> IMMEDIATE1_VALUE_SHIFT) as i32
    }
}

/// The value returned by this is expected to be cast to a *mut Term.
/// We do not do that here because we can't be sure the value will fit
/// in the native pointer width. It is up to the caller to do so
#[inline]
pub const fn unbox(term: u32) -> u32 {
    term & !(MASK_PRIMARY | MASK_LITERAL)
}

/// The value returned by this is expected to be cast to a *mut Cons.
/// We do not do that here because we can't be sure the value will fit
/// in the native pointer width. It is up to the caller to do so
#[inline]
pub const fn unbox_list(term: u32) -> u32 {
    term & !(MASK_PRIMARY | MASK_LITERAL)
}
