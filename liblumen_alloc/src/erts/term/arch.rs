mod constants_64;
mod constants_32;

pub mod arch32 {
    ///! This module exposes 32-bit architecture specific values and functions
    pub use super::constants_32::*;
}

pub mod arch64 {
    ///! This module exposes 64-bit architecture specific values and functions
    pub use super::constants_64::*;
}

pub mod native {
    ///! This module exposes architecture specific values and functions
    ///! for the current host architecture, depending on the configured
    ///! target pointer width
    use crate::erts::term::{Term, Cons};

    #[cfg(target_pointer_width = "32")]
    use super::constants_32 as constants;
    #[cfg(target_pointer_width = "64")]
    use super::constants_64 as constants;

    pub const NUM_BITS: usize = constants::NUM_BITS as usize;
    pub const MIN_ALIGNMENT: usize = constants::MIN_ALIGNMENT as usize;

    /// This is the highest assignable aligned address on this architecture
    pub const MAX_ALIGNED_ADDR: usize = constants::MAX_ALIGNED_ADDR as usize;

    pub const MAX_IMMEDIATE1_VALUE: usize = constants::MAX_IMMEDIATE1_VALUE as usize;
    pub const MAX_IMMEDIATE2_VALUE: usize = constants::MAX_IMMEDIATE2_VALUE as usize;

    pub const MIN_SMALLINT_VALUE: isize = constants::MIN_SMALLINT_VALUE as isize;
    pub const MAX_SMALLINT_VALUE: isize = constants::MAX_SMALLINT_VALUE as isize;

    pub const MAX_ATOM_ID: usize = constants::MAX_ATOM_ID as usize;

    // Primary types
    pub const FLAG_HEADER: usize = constants::FLAG_HEADER as usize;
    pub const FLAG_LIST: usize = constants::FLAG_LIST as usize;
    pub const FLAG_BOXED: usize = constants::FLAG_BOXED as usize;
    pub const FLAG_LITERAL: usize = constants::FLAG_LITERAL as usize;
    pub const FLAG_IMMEDIATE: usize = constants::FLAG_IMMEDIATE as usize;
    pub const FLAG_IMMEDIATE2: usize = constants::FLAG_IMMEDIATE2 as usize;
    // First class immediates
    pub const FLAG_PID: usize = constants::FLAG_PID as usize;
    pub const FLAG_PORT: usize = constants::FLAG_PORT as usize;
    pub const FLAG_SMALL_INTEGER: usize = constants::FLAG_SMALL_INTEGER as usize;
    // We store the sign for small ints in the highest of the immediate2 bits
    pub const FLAG_SMALL_INTEGER_SIGN: usize = constants::FLAG_SMALL_INTEGER_SIGN as usize;
    // Second class immediates
    pub const FLAG_ATOM: usize = constants::FLAG_ATOM as usize;
    pub const FLAG_CATCH: usize = constants::FLAG_CATCH as usize;
    pub const FLAG_UNUSED_1: usize = constants::FLAG_UNUSED_1 as usize;
    pub const FLAG_NIL: usize = constants::FLAG_NIL as usize;
    // Header types
    pub const FLAG_TUPLE: usize = constants::FLAG_TUPLE as usize;
    pub const FLAG_NONE: usize = constants::FLAG_NONE as usize;
    pub const FLAG_POS_BIG_INTEGER: usize = constants::FLAG_POS_BIG_INTEGER as usize;
    pub const FLAG_NEG_BIG_INTEGER: usize = constants::FLAG_NEG_BIG_INTEGER as usize;
    pub const FLAG_REFERENCE: usize = constants::FLAG_REFERENCE as usize;
    pub const FLAG_CLOSURE: usize = constants::FLAG_CLOSURE as usize;
    pub const FLAG_FLOAT: usize = constants::FLAG_FLOAT as usize;
    pub const FLAG_RESOURCE_REFERENCE: usize = constants::FLAG_RESOURCE_REFERENCE as usize;
    pub const FLAG_PROCBIN: usize = constants::FLAG_PROCBIN as usize;
    pub const FLAG_HEAPBIN: usize = constants::FLAG_HEAPBIN as usize;
    pub const FLAG_SUBBINARY: usize = constants::FLAG_SUBBINARY as usize;
    pub const FLAG_MATCH_CTX: usize = constants::FLAG_MATCH_CTX as usize;
    pub const FLAG_EXTERN_PID: usize = constants::FLAG_EXTERN_PID as usize;
    pub const FLAG_EXTERN_PORT: usize = constants::FLAG_EXTERN_PORT as usize;
    pub const FLAG_EXTERN_REF: usize = constants::FLAG_EXTERN_REF as usize;
    pub const FLAG_MAP: usize = constants::FLAG_MAP as usize;

    /// The pattern 0b0101 out to usize bits, but with the header bits
    /// masked out, and flagged as the none value
    pub const NONE_VAL: usize = constants::NONE_VAL as usize;

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn is_literal(term: usize) -> bool {
        constants::is_literal(term as u32)
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn is_literal(term: usize) -> bool {
        constants::is_literal(term as u64)
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn primary_tag(term: usize) -> usize {
        constants::primary_tag(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn primary_tag(term: usize) -> usize {
        constants::primary_tag(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn make_boxed<T>(value: *const T) -> usize {
        unsafe { value as usize | FLAG_BOXED }
    }

    #[inline]
    pub const fn make_boxed_literal<T>(value: *const T) -> usize {
        unsafe { value as usize | FLAG_BOXED | FLAG_LITERAL }
    }

    #[inline]
    pub const fn make_list(value: *const Cons) -> usize {
        unsafe { value as usize | FLAG_LIST }
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn make_header(term: usize, tag: usize) -> usize {
        constants::make_header(term as u32, tag as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn make_header(term: usize, tag: usize) -> usize {
        constants::make_header(term as u64, tag as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn header_tag(term: usize) -> usize {
        constants::header_tag(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn header_tag(term: usize) -> usize {
        constants::header_tag(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn header_arityval_tag(term: usize) -> usize {
        constants::header_arityval_tag(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn header_arityval_tag(term: usize) -> usize {
        constants::header_arityval_tag(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn header_value(term: usize) -> usize {
        constants::header_value(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn header_value(term: usize) -> usize {
        constants::header_value(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn make_immediate1(value: usize, tag: usize) -> usize {
        constants::make_immediate1(value as u32, tag as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn make_immediate1(value: usize, tag: usize) -> usize {
        constants::make_immediate1(value as u64, tag as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn make_immediate2(value: usize, tag: usize) -> usize {
        constants::make_immediate2(value as u32, tag as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn make_immediate2(value: usize, tag: usize) -> usize {
        constants::make_immediate2(value as u64, tag as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn immediate1_tag(term: usize) -> usize {
        constants::immediate1_tag(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn immediate1_tag(term: usize) -> usize {
        constants::immediate1_tag(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn immediate2_tag(term: usize) -> usize {
        constants::immediate2_tag(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn immediate2_tag(term: usize) -> usize {
        constants::immediate2_tag(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn immediate1_value(term: usize) -> usize {
        constants::immediate1_value(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn immediate1_value(term: usize) -> usize {
        constants::immediate1_value(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn immediate2_value(term: usize) -> usize {
        constants::immediate2_value(term as u32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn immediate2_value(term: usize) -> usize {
        constants::immediate2_value(term as u64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub fn make_smallint(value: isize) -> usize {
        debug_assert!(value <= MAX_SMALLINT_VALUE);
        debug_assert!(value >= MIN_SMALLINT_VALUE);
        constants::make_smallint(value as i32) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub fn make_smallint(value: isize) -> usize {
        debug_assert!(value <= MAX_SMALLINT_VALUE);
        debug_assert!(value >= MIN_SMALLINT_VALUE);
        constants::make_smallint(value as i64) as usize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub fn smallint_value(term: usize) -> isize {
        constants::smallint_value(term as u32) as isize
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub fn smallint_value(term: usize) -> isize {
        constants::smallint_value(term as u64) as isize
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn unbox(term: usize) -> *mut Term {
        constants::unbox(term as u32) as *mut Term
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn unbox(term: usize) -> *mut Term {
        constants::unbox(term as u64) as *mut Term
    }

    #[inline]
    #[cfg(target_pointer_width = "32")]
    pub const fn unbox_list(term: usize) -> *mut Cons {
        constants::unbox_list(term as u32) as *mut Cons
    }

    #[inline]
    #[cfg(target_pointer_width = "64")]
    pub const fn unbox_list(term: usize) -> *mut Cons {
        constants::unbox_list(term as u64) as *mut Cons
    }
}
