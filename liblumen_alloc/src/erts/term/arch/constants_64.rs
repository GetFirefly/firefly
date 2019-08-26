#![allow(unused)]
///! This module contains constants for 64-bit architectures used by the term
///! implementation.
///! On 64-bit platforms, the highest 16 bits of pointers are unused and
///! because we enforce 8-byte aligned allocations, the lowest 3 bits are
///! also unused. To keep things simple on this arch, we put all of our flags in the
///! highest 16 bits, with the exception of the literal flag, which we put
///! in the lowest bit of the address. This means that for pointer-typed terms,
///! the raw value just needs to be masked to access either the pointer or the flags,
///! no shifts are required.
use core::mem;
use crate::erts::term::Term;
use crate::erts::term::list::Cons;

pub const NUM_BITS: u64 = 64;
pub const MIN_ALIGNMENT: u64 = 8;

/// This is the highest logical 8-byte aligned address on this architecture
const MAX_LOGICAL_ALIGNED_ADDR: u64 = u64::max_value() & !(MIN_ALIGNMENT - 1);
/// This is the highest assignable 8-byte aligned address on this architecture
///
/// NOTE: On all modern 64-bit systems I'm aware of, the highest 16 bits are unused
pub const MAX_ALIGNED_ADDR: u64 = MAX_LOGICAL_ALIGNED_ADDR - (0xFFFFu64 << (NUM_BITS - 16));

const PRIMARY_SHIFT: u64 = NUM_BITS - 2;
const IMMEDIATE1_SHIFT: u64 = NUM_BITS - 4;
const IMMEDIATE2_SHIFT: u64 = NUM_BITS - 6;
const HEADER_TAG_SHIFT: u64 = NUM_BITS - 6;

// This is the largest value that will fit in a first-class immediate (i.e. pid)
// Integer immediates are handled specially, due to the extra bit required for signing
pub const MAX_IMMEDIATE1_VALUE: u64 = u64::max_value() >> (NUM_BITS - IMMEDIATE1_SHIFT);
// This is the largest value that will fit in a second-class immediate (i.e. atom)
pub const MAX_IMMEDIATE2_VALUE: u64 = u64::max_value() >> (NUM_BITS - IMMEDIATE2_SHIFT);
// Small integers require an extra bit for the sign, 5 bits, i.e. they are effectively 60bit
pub const MIN_SMALLINT_VALUE: i64 = i64::min_value() >> (NUM_BITS - IMMEDIATE1_SHIFT);
pub const MAX_SMALLINT_VALUE: i64 = i64::max_value() >> (NUM_BITS - IMMEDIATE1_SHIFT);
// Atom IDs are unsigned integer values and stored in a second-class immediate
pub const MAX_ATOM_ID: u64 = u64::max_value() >> (NUM_BITS - IMMEDIATE2_SHIFT);

// Primary types
pub const FLAG_HEADER: u64 = 0;
pub const FLAG_LIST: u64 = 1 << PRIMARY_SHIFT;
pub const FLAG_BOXED: u64 = 2 << PRIMARY_SHIFT;
// NOTE: This flag is only used with BOXED and LIST terms, and indicates that the term
// is a pointer to a literal, rather than a pointer to a term on the process heap/stack.
// Literals are stored as constants in the compiled code, so these terms are never GCed.
// In order to properly check if a term is a literal, you must first mask off the primary
// bits and verify it is either boxed or list, then mask off the immediate1 bits, and check
// for the literal tag
pub const FLAG_LITERAL: u64 = 1;
pub const FLAG_IMMEDIATE: u64 = 3 << PRIMARY_SHIFT;
pub const FLAG_IMMEDIATE2: u64 = FLAG_IMMEDIATE | (2 << IMMEDIATE1_SHIFT);
// First class immediates
pub const FLAG_PID: u64 = 0 | FLAG_IMMEDIATE;
pub const FLAG_PORT: u64 = (1 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE;
pub const FLAG_SMALL_INTEGER: u64 = (3 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE;
// We store the sign for small ints in the highest of the immediate2 bits
pub const FLAG_SMALL_INTEGER_SIGN: u64 = (1 << (IMMEDIATE2_SHIFT + 1));
// Second class immediates
pub const FLAG_ATOM: u64 = 0 | FLAG_IMMEDIATE2;
pub const FLAG_CATCH: u64 = (1 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
pub const FLAG_UNUSED_1: u64 = (2 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
pub const FLAG_NIL: u64 = (3 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
// Header types
pub const FLAG_TUPLE: u64 = 0 | FLAG_HEADER;
pub const FLAG_NONE: u64 = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_POS_BIG_INTEGER: u64 = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_NEG_BIG_INTEGER: u64 = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_REFERENCE: u64 = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_CLOSURE: u64 = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_FLOAT: u64 = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_RESOURCE_REFERENCE: u64 = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_PROCBIN: u64 = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_HEAPBIN: u64 = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_SUBBINARY: u64 = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_MATCH_CTX: u64 = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_EXTERN_PID: u64 = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_EXTERN_PORT: u64 = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_EXTERN_REF: u64 = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER;
pub const FLAG_MAP: u64 = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER;

// The primary tag is given by masking bits 0-2
pub const MASK_PRIMARY: u64 = 0xC000_0000_0000_0000;
// The literal tag is given by masking the lowest bit
pub const MASK_LITERAL: u64 = 0x1;
// First class immediate tags are given by masking bits 2-4
pub const MASK_IMMEDIATE1_TAG: u64 = 0x3000_0000_0000_0000;
// Second class immediate tags are given by masking bits 4-6
pub const MASK_IMMEDIATE2_TAG: u64 = 0x0C00_0000_0000_0000;
// To mask off the entire immediate header, we mask off both the primary and immediate tag
pub const MASK_IMMEDIATE1: u64 = MASK_PRIMARY | MASK_IMMEDIATE1_TAG;
pub const MASK_IMMEDIATE2: u64 = MASK_IMMEDIATE1 | MASK_IMMEDIATE2_TAG;
// Header is composed of 2 primary tag bits, and 4 subtag bits:
pub const MASK_HEADER: u64 = MASK_HEADER_PRIMARY | MASK_HEADER_ARITYVAL;
// The primary tag is used to identify that a word is a header
pub const MASK_HEADER_PRIMARY: u64 = MASK_PRIMARY;
// The arityval is a subtag that identifies the boxed type
// This value is used as a marker in some checks, but it is essentially equivalent
// to `FLAG_TUPLE & !FLAG_HEADER`, which is simply the value 0
pub const ARITYVAL: u64 = 0;
// The following is a mask for the actual arityval value
pub const MASK_HEADER_ARITYVAL: u64 = 0x3C00_0000_0000_0000;

/// The pattern 0b0101 out to usize bits, but with the header bits
/// masked out, and flagged as the none value
pub const NONE_VAL: u64 = 0x155555554AAAAAAAu64 & !MASK_HEADER;

#[inline]
pub const fn is_literal(term: u64) -> bool {
    term & FLAG_LITERAL == FLAG_LITERAL
}

#[inline]
pub const fn primary_tag(term: u64) -> u64 {
    term & MASK_PRIMARY
}

#[inline]
pub const fn make_header(term: u64, tag: u64) -> u64 {
    term | tag
}

#[inline]
pub const fn header_tag(term: u64) -> u64 {
    term & MASK_HEADER
}

#[inline]
pub const fn header_arityval_tag(term: u64) -> u64 {
    term & MASK_HEADER_ARITYVAL
}

#[inline]
pub const fn header_value(term: u64) -> u64 {
    term & !MASK_HEADER
}

#[inline]
pub const fn make_immediate1(value: u64, tag: u64) -> u64 {
    value | tag
}

#[inline]
pub const fn make_immediate2(value: u64, tag: u64) -> u64 {
    value | tag
}

#[inline]
pub const fn immediate1_tag(term: u64) -> u64 {
    term & MASK_IMMEDIATE1
}

#[inline]
pub const fn immediate2_tag(term: u64) -> u64 {
    term & MASK_IMMEDIATE2
}

#[inline]
pub const fn immediate1_value(term: u64) -> u64 {
    term & !MASK_IMMEDIATE1
}

#[inline]
pub const fn immediate2_value(term: u64) -> u64 {
    term & !MASK_IMMEDIATE2
}

#[inline]
pub fn make_smallint(value: i64) -> u64 {
    debug_assert!(value <= MAX_SMALLINT_VALUE);
    debug_assert!(value >= MIN_SMALLINT_VALUE);
    match value.signum() {
        0 | 1 => (value as u64) | FLAG_SMALL_INTEGER,
        _ => (value as u64) | FLAG_SMALL_INTEGER_SIGN | FLAG_SMALL_INTEGER
    }
}

#[inline]
pub fn smallint_value(term: u64) -> i64 {
    if term & FLAG_SMALL_INTEGER_SIGN == FLAG_SMALL_INTEGER_SIGN {
        // Signed, i.e. negative
        // In this case we don't have to do anything, the high
        // bits must always remain set to enforce the min value,
        // which is already a given by FLAG_SMALL_INTEGER | FLAG_SMALL_INTEGER_SIGN
        term as i64
    } else {
        // Unsigned, i.e. positive or zero
        // In this case we have to strip the high bits, as they
        // must always remain unset to enforce the max value
        (term & !(FLAG_SMALL_INTEGER | FLAG_SMALL_INTEGER_SIGN)) as i64
    }
}

/// The value returned by this is expected to be cast to a *mut Term.
/// We do not do that here because we can't be sure the value will fit
/// in the native pointer width. It is up to the caller to do so
#[inline]
pub const fn unbox(term: u64) -> u64 {
    term & !(MASK_PRIMARY | MASK_LITERAL)
}

/// The value returned by this is expected to be cast to a *mut Cons.
/// We do not do that here because we can't be sure the value will fit
/// in the native pointer width. It is up to the caller to do so
#[inline]
pub const fn unbox_list(term: u64) -> u64 {
    term & !(MASK_PRIMARY | MASK_LITERAL)
}
