#![allow(unused_parens)]

use core::cmp;
use core::convert::TryInto;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr;

use alloc::string::String;

use crate::borrow::CloneToProcess;
use crate::erts::exception::runtime;
use crate::erts::exception::system::Alloc;
use crate::erts::term::binary::aligned_binary::AlignedBinary;
use crate::erts::term::resource;
use crate::erts::term::InvalidTermError;
use crate::erts::ProcessControlBlock;

use num_bigint::BigInt;

use super::*;

#[cfg(target_pointer_width = "64")]
mod constants {
    use super::Term;

    use crate::erts::term::list::Cons;
    use crate::mem::bit_size_of;

    ///! This module contains constants for 64-bit architectures used by the term
    /// implementation. !
    ///! On 64-bit platforms, the highest 16 bits of pointers are unused and
    ///! because we enforce 8-byte aligned allocations, the lowest 3 bits are
    ///! also unused. To keep things simple on this arch, we put all of our flags in the
    ///! highest 16 bits, with the exception of the literal flag, which we put
    ///! in the lowest bit of the address. This means that for pointer-typed terms,
    ///! the raw value just needs to be masked to access either the pointer or the flags,
    ///! no shifts are required.
    #[cfg(test)]
    use core::mem;

    const NUM_BITS: usize = bit_size_of::<usize>();
    #[cfg(test)]
    const MIN_ALIGNMENT: usize = mem::align_of::<usize>();

    /// This is the highest logical 8-byte aligned address on this architecture
    #[cfg(test)]
    const MAX_LOGICAL_ALIGNED_ADDR: usize = usize::max_value() & !(MIN_ALIGNMENT - 1);
    /// This is the highest assignable 8-byte aligned address on this architecture
    ///
    /// NOTE: On all modern 64-bit systems I'm aware of, the highest 16 bits are unused
    #[cfg(test)]
    pub const MAX_ALIGNED_ADDR: usize = MAX_LOGICAL_ALIGNED_ADDR - (0xFFFFusize << (NUM_BITS - 16));

    pub const MAX_IMMEDIATE1_VALUE: usize = usize::max_value() >> 4;
    pub const MAX_IMMEDIATE2_VALUE: usize = usize::max_value() >> 6;

    const PRIMARY_SHIFT: usize = NUM_BITS - 2;
    const IMMEDIATE1_SHIFT: usize = NUM_BITS - 4;
    const IMMEDIATE2_SHIFT: usize = NUM_BITS - 6;
    const HEADER_TAG_SHIFT: usize = NUM_BITS - 6;

    // Primary types
    pub const FLAG_HEADER: usize = 0;
    pub const FLAG_LIST: usize = 1 << PRIMARY_SHIFT;
    pub const FLAG_BOXED: usize = 2 << PRIMARY_SHIFT;
    pub const FLAG_IMMEDIATE: usize = 3 << PRIMARY_SHIFT;
    pub const FLAG_IMMEDIATE2: usize = FLAG_IMMEDIATE | (2 << IMMEDIATE1_SHIFT);

    // First class immediates
    pub const FLAG_PID: usize = (0 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE;
    pub const FLAG_PORT: usize = (1 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE;
    pub const FLAG_SMALL_INTEGER: usize = (3 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE;

    // Small Integers are isize, but with the loss of the FLAG_SMALL_INTEGER bits
    pub const MIN_SMALLINT_VALUE: isize = std::isize::MIN >> (NUM_BITS - IMMEDIATE1_SHIFT);
    pub const MAX_SMALLINT_VALUE: isize = std::isize::MAX >> (NUM_BITS - IMMEDIATE1_SHIFT);

    // Second class immediates
    pub const FLAG_ATOM: usize = (0 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
    pub const FLAG_CATCH: usize = (1 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
    /// Also unused in BEAM
    pub const FLAG_UNUSED_1: usize = (2 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;
    pub const FLAG_NIL: usize = (3 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2;

    // NOTE: This flag is only used with BOXED and LIST terms, and indicates that the term
    // is a pointer to a literal, rather than a pointer to a term on the process heap/stack.
    // Literals are stored as constants in the compiled code, so these terms are never GCed.
    // In order to properly check if a term is a literal, you must first mask off the primary
    // bits and verify it is either boxed or list, then mask off the immediate1 bits, and check
    // for the literal tag
    pub const FLAG_LITERAL: usize = 1;

    // Header types
    pub const FLAG_TUPLE: usize = (0 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    /// Binary aggregate in BEAM
    pub const FLAG_NONE: usize = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_POS_BIG_INTEGER: usize = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_NEG_BIG_INTEGER: usize = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_REFERENCE: usize = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_CLOSURE: usize = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_FLOAT: usize = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    /// EXPORT in BEAM
    pub const FLAG_RESOURCE_REFERENCE: usize = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_PROCBIN: usize = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_HEAPBIN: usize = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_SUBBINARY: usize = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_MATCH_CTX: usize = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_EXTERN_PID: usize = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_EXTERN_PORT: usize = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_EXTERN_REF: usize = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER;
    pub const FLAG_MAP: usize = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER;

    // The primary tag is given by masking bits 0-2
    pub const MASK_PRIMARY: usize = 0xC000_0000_0000_0000;
    // First class immediate tags are given by masking bits 2-4
    pub const MASK_IMMEDIATE1_TAG: usize = 0x3000_0000_0000_0000;
    // Second class immediate tags are given by masking bits 4-6
    pub const MASK_IMMEDIATE2_TAG: usize = 0x0C00_0000_0000_0000;
    // To mask off the entire immediate header, we mask off both the primary and immediate tag
    pub const MASK_IMMEDIATE1: usize = MASK_PRIMARY | MASK_IMMEDIATE1_TAG;
    pub const MASK_IMMEDIATE2: usize = MASK_IMMEDIATE1 | MASK_IMMEDIATE2_TAG;
    pub const MASK_HEADER_TAG: usize = 0x3C00_0000_0000_0000;
    // Header is composed of 2 primary tag bits, and 4 subtag bits:
    pub const MASK_HEADER: usize = MASK_PRIMARY | MASK_HEADER_TAG;

    // The literal tag is given by masking the lowest bit
    pub const MASK_LITERAL: usize = 0x1;

    /// The pattern 0b0101 out to usize bits, but with the header bits
    /// masked out, and flagged as the none value
    pub const NONE_VAL: usize = 0x155555554AAAAAAAusize & !MASK_HEADER;

    #[inline]
    pub const fn is_literal(term: usize) -> bool {
        term & FLAG_LITERAL == FLAG_LITERAL
    }

    #[inline]
    pub const fn primary_tag(term: usize) -> usize {
        term & MASK_PRIMARY
    }

    #[inline]
    pub const fn make_list(value: *const Cons) -> usize {
        unsafe { value as usize | FLAG_LIST }
    }

    #[inline]
    pub const fn make_header(term: usize, tag: usize) -> usize {
        term | tag
    }

    #[inline]
    pub const fn header_tag(term: usize) -> usize {
        term & MASK_HEADER
    }

    #[inline]
    pub const fn header_value(term: usize) -> usize {
        term & !MASK_HEADER
    }

    #[inline]
    pub const fn make_immediate1(value: usize, tag: usize) -> usize {
        value | tag
    }

    pub const fn make_smallint(value: isize) -> usize {
        FLAG_SMALL_INTEGER | (value as usize)
    }

    #[inline]
    pub const fn make_immediate2(value: usize, tag: usize) -> usize {
        value | tag
    }

    #[inline]
    pub const fn immediate1_tag(term: usize) -> usize {
        term & MASK_IMMEDIATE1
    }

    #[inline]
    pub const fn immediate2_tag(term: usize) -> usize {
        term & MASK_IMMEDIATE2
    }

    #[inline]
    pub const fn immediate1_value(term: usize) -> usize {
        term & !MASK_IMMEDIATE1
    }

    #[inline]
    pub const fn immediate2_value(term: usize) -> usize {
        term & !MASK_IMMEDIATE2
    }

    #[inline]
    pub const fn boxed_value(term: usize) -> *mut Term {
        (term & !(MASK_PRIMARY | MASK_LITERAL)) as *mut Term
    }

    #[inline]
    pub const fn list_value(term: usize) -> *mut Cons {
        (term & !(MASK_PRIMARY | MASK_LITERAL)) as *mut Cons
    }

    pub const fn smallint_value(term: usize) -> isize {
        (((term & !MASK_IMMEDIATE1) << (NUM_BITS - IMMEDIATE1_SHIFT)) as isize)
            >> (NUM_BITS - IMMEDIATE1_SHIFT)
    }
}

#[cfg(target_pointer_width = "32")]
mod constants {
    use super::Term;

    use crate::erts::term::list::Cons;
    use crate::mem::bit_size_of;
    ///! This module contains constants for 64-bit architectures used by the term
    /// implementation. !
    ///! On 32-bit platforms we generally can use the high bits like we
    ///! do on 64-bit platforms, as the full address space is rarely available,
    ///! e.g. Windows with 2-3gb addressable range, Linux with 3gb. But to avoid
    ///! the reliance on that fact, we use the low bits on 32-bit platforms
    ///! for the pointer-typed terms. We have 3 bits available to us because we
    ///! require an 8-byte minimum alignment for all allocations, ensuring that
    ///! the lowest 3 bits are always zeroes, and thus available for tags. For
    ///! non-pointer terms, the flags all go in the high bits, to make accessing
    ///! the value and tags as easy as applying a mask, no shifts needed.
    #[cfg(test)]
    use core::mem;

    const NUM_BITS: usize = bit_size_of::<usize>();
    #[cfg(test)]
    const MIN_ALIGNMENT: usize = mem::align_of::<usize>();

    #[cfg(test)]
    const MAX_LOGICAL_ALIGNED_ADDR: usize = usize::max_value() & !(MIN_ALIGNMENT - 1);
    /// This is the highest assignable 8-byte aligned address on this architecture
    ///
    /// NOTE: On Windows and Linux, this address is actually capped at 2gb and 3gb respectively,
    /// but other platforms may not have this restriction, so our tagging scheme tries to avoid
    /// using high bits if at all possible. We do use the highest bit as a flag for literal values,
    /// i.e. pointers to literal constants
    #[cfg(test)]
    pub const MAX_ALIGNED_ADDR: usize = MAX_LOGICAL_ALIGNED_ADDR - (1usize << (NUM_BITS - 1));

    // https://blog.stenmans.org/theBeamBook/#_the_tagging_scheme

    const PRIMARY_SHIFT: usize = 0;
    const IMMEDIATE1_SHIFT: usize = 2;
    const IMMEDIATE1_VALUE_SHIFT: usize = 4;
    const IMMEDIATE2_SHIFT: usize = 4;
    const IMMEDIATE2_VALUE_SHIFT: usize = 6;
    const HEADER_TAG_SHIFT: usize = 2;
    const HEADER_VALUE_SHIFT: usize = 6;

    pub const MAX_IMMEDIATE1_VALUE: usize = usize::max_value() >> IMMEDIATE1_VALUE_SHIFT;
    pub const MAX_IMMEDIATE2_VALUE: usize = usize::max_value() >> IMMEDIATE2_VALUE_SHIFT;

    const FLAG_PRIMARY: usize = 0;

    // Primary types
    pub const FLAG_HEADER: usize = (0 << PRIMARY_SHIFT) | FLAG_PRIMARY; // 0b00
    pub const FLAG_LIST: usize = (1 << PRIMARY_SHIFT) | FLAG_PRIMARY; // 0b01
    pub const FLAG_BOXED: usize = (2 << PRIMARY_SHIFT) | FLAG_PRIMARY; // 0b10
    pub const FLAG_IMMEDIATE: usize = (3 << PRIMARY_SHIFT) | FLAG_PRIMARY; // 0b11

    // First class immediates
    pub const FLAG_PID: usize = (0 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b00_11
    pub const FLAG_PORT: usize = (1 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b01_011
    pub const FLAG_IMMEDIATE2: usize = (2 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b10_11
    pub const FLAG_SMALL_INTEGER: usize = (3 << IMMEDIATE1_SHIFT) | FLAG_IMMEDIATE; // 0b11_11

    // Small Integers are isize, but with the loss of the FLAG_SMALL_INTEGER bits
    pub const MIN_SMALLINT_VALUE: isize = std::isize::MIN >> IMMEDIATE1_VALUE_SHIFT;
    pub const MAX_SMALLINT_VALUE: isize = std::isize::MAX >> IMMEDIATE1_VALUE_SHIFT;

    // Second class immediates
    pub const FLAG_ATOM: usize = (0 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b00_10_11
    pub const FLAG_CATCH: usize = (1 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b01_10_11
    /// Also unused in BEAM
    pub const FLAG_UNUSED_1: usize = (2 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b10_10_11
    pub const FLAG_NIL: usize = (3 << IMMEDIATE2_SHIFT) | FLAG_IMMEDIATE2; // 0b11_10_11

    pub const FLAG_LITERAL: usize = 1 << (NUM_BITS - 1); // 0b1000_0000_0000_0000_0000_0000_0000_0000

    // Header types, these flags re-use the literal flag bit, as it only applies to primary tags
    pub const FLAG_TUPLE: usize = (0 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0000_00
    /// Binary aggregate in BEAM
    pub const FLAG_NONE: usize = (1 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0001_00
    pub const FLAG_POS_BIG_INTEGER: usize = (2 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0010_00
    pub const FLAG_NEG_BIG_INTEGER: usize = (3 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0011_00
    pub const FLAG_REFERENCE: usize = (4 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0100_00
    pub const FLAG_CLOSURE: usize = (5 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0101_00
    pub const FLAG_FLOAT: usize = (6 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0110_00
    /// EXPORT in BEAM
    pub const FLAG_RESOURCE_REFERENCE: usize = (7 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b0111_00
    pub const FLAG_PROCBIN: usize = (8 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1000_00
    pub const FLAG_HEAPBIN: usize = (9 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1001_00
    pub const FLAG_SUBBINARY: usize = (10 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1010_00
    pub const FLAG_MATCH_CTX: usize = (11 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1011_00
    pub const FLAG_EXTERN_PID: usize = (12 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1100_00
    pub const FLAG_EXTERN_PORT: usize = (13 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1101_00
    pub const FLAG_EXTERN_REF: usize = (14 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1110_00
    pub const FLAG_MAP: usize = (15 << HEADER_TAG_SHIFT) | FLAG_HEADER; // 0b1111_00

    // The primary tag is given by masking bits 0-1
    pub const MASK_PRIMARY: usize = 0b11;
    // First class immediate tags are given by masking bits 2-3
    pub const MASK_IMMEDIATE1_TAG: usize = 0b11_00;
    // Second class immediate tags are given by masking bits 6-7
    pub const MASK_IMMEDIATE2_TAG: usize = 0b11_00_00;
    // To mask off the entire immediate header, we mask off the primary and immediate tag
    pub const MASK_IMMEDIATE1: usize = MASK_IMMEDIATE1_TAG | MASK_PRIMARY;
    pub const MASK_IMMEDIATE2: usize = MASK_IMMEDIATE2_TAG | MASK_IMMEDIATE1;
    pub const MASK_HEADER_TAG: usize = 0b1111_00;
    pub const MASK_HEADER: usize = MASK_HEADER_TAG | MASK_PRIMARY;

    // The literal flag is the highest bit.  Assert check that we're not hitting it under large
    // memory usage.
    pub const MASK_LITERAL: usize = 1 << (NUM_BITS - 1);

    /// The pattern 0b0101 out to usize bits, but with the header bits
    /// masked out, and flagged as the none value
    pub const NONE_VAL: usize = 0x55555555usize & !MASK_HEADER;

    #[inline]
    pub const fn is_literal(term: usize) -> bool {
        term & FLAG_LITERAL == FLAG_LITERAL
    }

    #[inline]
    pub const fn primary_tag(term: usize) -> usize {
        term & MASK_PRIMARY
    }

    #[inline]
    pub const fn make_list(value: *const Cons) -> usize {
        unsafe { value as usize | FLAG_LIST }
    }

    #[inline]
    pub const fn make_header(term: usize, tag: usize) -> usize {
        (term << HEADER_VALUE_SHIFT) | tag
    }

    #[inline]
    pub const fn header_tag(term: usize) -> usize {
        term & MASK_HEADER
    }

    #[inline]
    pub const fn header_value(term: usize) -> usize {
        (term & !MASK_HEADER) >> HEADER_VALUE_SHIFT
    }

    #[inline]
    pub const fn make_immediate1(value: usize, tag: usize) -> usize {
        (value << IMMEDIATE1_VALUE_SHIFT) | tag
    }

    pub const fn make_smallint(value: isize) -> usize {
        ((value << IMMEDIATE1_VALUE_SHIFT) as usize) | FLAG_SMALL_INTEGER
    }

    #[inline]
    pub const fn make_immediate2(value: usize, tag: usize) -> usize {
        (value << IMMEDIATE2_VALUE_SHIFT) | tag
    }

    #[inline]
    pub const fn immediate1_tag(term: usize) -> usize {
        term & MASK_IMMEDIATE1
    }

    #[inline]
    pub const fn immediate2_tag(term: usize) -> usize {
        term & MASK_IMMEDIATE2
    }

    #[inline]
    pub const fn immediate1_value(term: usize) -> usize {
        (term & !MASK_IMMEDIATE1) >> IMMEDIATE1_VALUE_SHIFT
    }

    #[inline]
    pub const fn immediate2_value(term: usize) -> usize {
        (term & !MASK_IMMEDIATE2) >> IMMEDIATE2_VALUE_SHIFT
    }

    #[inline]
    pub const fn boxed_value(term: usize) -> *mut Term {
        (term & !(MASK_PRIMARY | MASK_LITERAL)) as *mut Term
    }

    #[inline]
    pub const fn list_value(term: usize) -> *mut Cons {
        (term & !(MASK_PRIMARY | MASK_LITERAL)) as *mut Cons
    }

    pub const fn smallint_value(term: usize) -> isize {
        (
            // cast to isize before shift to get sign extension
            (term & !MASK_IMMEDIATE1) as isize
        ) >> IMMEDIATE1_VALUE_SHIFT
    }
}

mod typecheck {
    ///! This module contains functions which perform type checking on raw term values.
    ///!
    ///! These functions are all constant functions, so they are also used in static
    /// assertions ! to verify that the functions are correct at compile-time, rather than
    /// depending on tests.
    use super::constants;

    pub const NONE: usize = constants::NONE_VAL | constants::FLAG_NONE;
    pub const NIL: usize = constants::FLAG_NIL;

    /// Returns true if this term is the none value
    #[inline]
    pub const fn is_none(term: usize) -> bool {
        term == NONE
    }

    /// Returns true if this term is nil
    #[inline]
    pub const fn is_nil(term: usize) -> bool {
        term == NIL
    }

    /// Returns true if this term is a literal
    #[inline]
    pub fn is_literal(term: usize) -> bool {
        (is_boxed(term) || is_list(term)) && constants::is_literal(term)
    }

    /// Returns true if this term is a list
    #[inline]
    pub fn is_list(term: usize) -> bool {
        constants::primary_tag(term) == constants::FLAG_LIST
    }

    /// Returns true if this term is an atom
    #[inline]
    pub fn is_atom(term: usize) -> bool {
        is_immediate2(term) && constants::immediate2_tag(term) == constants::FLAG_ATOM
    }

    /// Returns true if this term is a small integer (i.e. fits in a usize)
    #[inline]
    pub fn is_smallint(term: usize) -> bool {
        is_immediate(term) && constants::immediate1_tag(term) == constants::FLAG_SMALL_INTEGER
    }

    /// Returns true if this term is a big integer (i.e. arbitrarily large)
    #[inline]
    pub fn is_bigint(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_POS_BIG_INTEGER
            || constants::header_tag(term) == constants::FLAG_NEG_BIG_INTEGER
    }

    /// Returns true fi this term is a small or big integer
    #[cfg(test)]
    pub fn is_integer(term: usize) -> bool {
        is_smallint(term) || is_bigint(term)
    }

    /// Returns true if this term is a float
    pub const fn is_float(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_FLOAT
    }

    /// Returns true fi this term is a small integer, big integer, or float.
    #[cfg(test)]
    pub fn is_number(term: usize) -> bool {
        is_integer(term) || is_float(term)
    }

    pub const fn is_function(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_CLOSURE
    }

    /// Returns true if this term is a tuple
    #[inline]
    pub const fn is_tuple(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_TUPLE
    }

    /// Returns true if this term is a map
    #[inline]
    pub const fn is_map(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_MAP
    }

    /// Returns true if this term is a pid on the local node
    #[inline]
    pub fn is_local_pid(term: usize) -> bool {
        is_immediate(term) && (constants::immediate1_tag(term) == constants::FLAG_PID)
    }

    /// Returns true if this term is a pid on some other node
    #[inline]
    pub const fn is_remote_pid(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_EXTERN_PID
    }

    /// Returns true if this term is a pid on the locale node or some other node
    #[cfg(test)]
    pub fn is_pid(term: usize) -> bool {
        is_local_pid(term) || is_remote_pid(term)
    }

    /// Returns true if this term is a pid
    #[inline]
    pub fn is_port(term: usize) -> bool {
        is_local_port(term) || is_remote_port(term)
    }

    /// Returns true if this term is a port on the local node
    #[inline]
    pub fn is_local_port(term: usize) -> bool {
        is_immediate(term) && constants::immediate1_tag(term) == constants::FLAG_PORT
    }

    /// Returns true if this term is a port on some other node
    #[inline]
    pub const fn is_remote_port(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_EXTERN_PORT
    }

    /// Returns true if this term is a reference
    #[inline]
    pub fn is_reference(term: usize) -> bool {
        is_local_reference(term) || is_remote_reference(term)
    }

    /// Returns true if this term is a reference on the local node
    #[inline]
    pub const fn is_local_reference(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_REFERENCE
    }

    /// Returns true if this term is a reference on some other node
    #[inline]
    pub const fn is_remote_reference(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_EXTERN_REF
    }

    pub const fn is_resource_reference(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_RESOURCE_REFERENCE
    }

    /// Returns true if this term is a bitstring
    #[cfg(test)]
    pub fn is_bitstring(term: usize) -> bool {
        is_heapbin(term) || is_match_context(term) || is_procbin(term) || is_subbinary(term)
    }

    /// Returns true if this term is a reference-counted binary
    #[inline]
    pub const fn is_procbin(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_PROCBIN
    }

    /// Returns true if this term is a binary on a process heap
    #[inline]
    pub const fn is_heapbin(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_HEAPBIN
    }

    /// Returns true if this term is a sub-binary reference
    #[inline]
    pub const fn is_subbinary(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_SUBBINARY
    }

    /// Returns true if this term is a binary match context
    #[inline]
    pub const fn is_match_context(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_MATCH_CTX
    }

    /// Returns true if this term is a closure
    #[inline]
    pub const fn is_closure(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_CLOSURE
    }

    /// Returns true if this term is a catch flag
    #[inline]
    pub const fn is_catch(term: usize) -> bool {
        constants::header_tag(term) == constants::FLAG_CATCH
    }

    /// Returns true if this term is an immediate value
    #[inline]
    pub const fn is_immediate(term: usize) -> bool {
        constants::primary_tag(term) == constants::FLAG_IMMEDIATE
    }

    #[inline]
    fn is_immediate2(term: usize) -> bool {
        is_immediate(term) && constants::immediate1_tag(term) == constants::FLAG_IMMEDIATE2
    }

    /// Returns true if this term is a constant value
    ///
    /// NOTE: Currently the meaning of constant in this context is
    /// equivalent to that of immediates, i.e. an immediate is a constant,
    /// and only immediates are constants. Realistically we should be able
    /// to support constants of arbitrary term type, but this is derived
    /// from BEAM for now
    #[inline]
    pub const fn is_const(term: usize) -> bool {
        is_immediate(term)
    }

    /// Returns true if this term is a boxed pointer
    #[inline]
    pub const fn is_boxed(term: usize) -> bool {
        constants::primary_tag(term) == constants::FLAG_BOXED
    }

    /// Returns true if this term is a header
    #[inline]
    pub const fn is_header(term: usize) -> bool {
        constants::primary_tag(term) == constants::FLAG_HEADER
    }
}

/// This struct is a general tagged pointer type for Erlang terms.
///
/// It is generally equivalent to a reference, with the exception of
/// the class of terms called "immediates", which are not boxed values,
/// but stored in the tagged value itself.
///
/// For immediates, the high 6 bits of the value are used for tags, while
/// boxed values use the high 2 bits as tag, and point to either another box,
/// the NONE value, or a header word. The header word leaves the high 2 bits
/// zeroed, and the next 4 high bits (arityval) as tag for the type of object
/// the header is for. In some cases the header contains part of the value, in
/// others the value begins immediately following the header word.
///
/// The raw term value is tagged, so it is not something you should access directly
/// unless you know what you are doing. Most of the time use of that value is left
/// to the internals of `Term` itself.
///
/// Since `Term` values are often pointers, it should be given the same considerations
/// that you would give a raw pointer/reference anywhere else
#[derive(Clone, Copy, Eq)]
#[repr(transparent)]
pub struct Term(usize);
impl Term {
    pub const MAX_IMMEDIATE1_VALUE: usize = constants::MAX_IMMEDIATE1_VALUE;
    pub const MAX_IMMEDIATE2_VALUE: usize = constants::MAX_IMMEDIATE2_VALUE;

    // Re-exported constants
    pub const FLAG_HEADER: usize = constants::FLAG_HEADER;
    pub const FLAG_LIST: usize = constants::FLAG_LIST;
    pub const FLAG_BOXED: usize = constants::FLAG_BOXED;
    pub const FLAG_LITERAL: usize = constants::FLAG_LITERAL;
    pub const FLAG_IMMEDIATE: usize = constants::FLAG_IMMEDIATE;
    pub const FLAG_IMMEDIATE2: usize = constants::FLAG_IMMEDIATE2;

    // First class immediates
    pub const FLAG_PID: usize = constants::FLAG_PID;
    pub const FLAG_PORT: usize = constants::FLAG_PORT;
    pub const FLAG_SMALL_INTEGER: usize = constants::FLAG_SMALL_INTEGER;

    pub const MIN_SMALLINT_VALUE: isize = constants::MIN_SMALLINT_VALUE;
    pub const MAX_SMALLINT_VALUE: isize = constants::MAX_SMALLINT_VALUE;

    // Second class immediates
    pub const FLAG_ATOM: usize = constants::FLAG_ATOM;
    pub const FLAG_CATCH: usize = constants::FLAG_CATCH;
    pub const FLAG_UNUSED_1: usize = constants::FLAG_UNUSED_1;
    pub const FLAG_NIL: usize = constants::FLAG_NIL;

    // Header types
    pub const FLAG_TUPLE: usize = constants::FLAG_TUPLE;
    pub const FLAG_NONE: usize = constants::FLAG_NONE;
    pub const FLAG_POS_BIG_INTEGER: usize = constants::FLAG_POS_BIG_INTEGER;
    pub const FLAG_NEG_BIG_INTEGER: usize = constants::FLAG_NEG_BIG_INTEGER;
    pub const FLAG_REFERENCE: usize = constants::FLAG_REFERENCE;
    pub const FLAG_CLOSURE: usize = constants::FLAG_CLOSURE;
    pub const FLAG_FLOAT: usize = constants::FLAG_FLOAT;
    pub const FLAG_RESOURCE_REFERENCE: usize = constants::FLAG_RESOURCE_REFERENCE;
    pub const FLAG_PROCBIN: usize = constants::FLAG_PROCBIN;
    pub const FLAG_HEAPBIN: usize = constants::FLAG_HEAPBIN;
    pub const FLAG_SUBBINARY: usize = constants::FLAG_SUBBINARY;
    pub const FLAG_MATCH_CTX: usize = constants::FLAG_MATCH_CTX;
    pub const FLAG_EXTERN_PID: usize = constants::FLAG_EXTERN_PID;
    pub const FLAG_EXTERN_PORT: usize = constants::FLAG_EXTERN_PORT;
    pub const FLAG_EXTERN_REF: usize = constants::FLAG_EXTERN_REF;
    pub const FLAG_MAP: usize = constants::FLAG_MAP;

    /// Used to represent the absence of any meaningful value, in particular
    /// it is used by the process heap allocator/garbage collector
    pub const NONE: Self = Self(typecheck::NONE);
    /// Represents the singleton nil value
    pub const NIL: Self = Self(typecheck::NIL);
    /// Represents the catch flag
    pub const CATCH: Self = Self(Self::FLAG_CATCH);

    /// Creates a header term from an arity and a tag (e.g. FLAG_PROCBIN)
    ///
    /// The `arity` is the number of _extra_ `core::mem::size_of::<Term>` that struct takes.  It
    /// *DOES NOT* include the `core::mem::size_of::<Term>` size of this header itself.
    #[inline]
    pub const fn make_header(arity: usize, tag: usize) -> Self {
        Self(constants::make_header(arity, tag))
    }

    /// Creates a boxed term from a pointer to the inner term
    pub fn make_boxed<T>(ptr: *const T) -> Self {
        let address = ptr as usize;

        assert_eq!(
            address & Self::FLAG_BOXED,
            0,
            "Pointer bits ({:032b}) colliding with boxed flag ({:032b})",
            address,
            Self::FLAG_BOXED
        );

        Self(address | Self::FLAG_BOXED)
    }

    /// Creates a boxed literal term from a pointer to the inner term
    #[inline]
    pub fn make_boxed_literal<T>(ptr: *const T) -> Self {
        let address = ptr as usize;

        assert_eq!(
            address & Self::FLAG_BOXED,
            0,
            "Pointer bits ({:032b}) colliding with boxed flag ({:032b})",
            address,
            Self::FLAG_BOXED
        );
        assert_eq!(
            address & Self::FLAG_LITERAL,
            0,
            "Pointer bits ({:032b}) colliding with literal flag ({:032b})",
            address,
            Self::FLAG_LITERAL
        );

        Self(Self::FLAG_LITERAL | address | Self::FLAG_BOXED)
    }

    /// Creates a list term from a pointer to a cons cell
    #[inline]
    pub const fn make_list(value: *const Cons) -> Self {
        Self(constants::make_list(value))
    }

    /// Creates a (local) pid value from a raw usize value
    #[inline]
    pub fn make_pid(serial_number: usize) -> Self {
        assert!(serial_number <= constants::MAX_IMMEDIATE1_VALUE);
        Self(constants::make_immediate1(serial_number, Self::FLAG_PID))
    }

    /// Creates a (local) port value from a raw usize value
    #[inline]
    pub fn make_port(value: usize) -> Self {
        assert!(value <= constants::MAX_IMMEDIATE1_VALUE);
        Self(constants::make_immediate1(value, Self::FLAG_PORT))
    }

    /// Creates a small integer term from a raw usize value
    #[inline]
    pub fn make_smallint(value: isize) -> Self {
        assert!(constants::MIN_SMALLINT_VALUE <= value);
        assert!(value <= constants::MAX_SMALLINT_VALUE);

        Self(constants::make_smallint(value))
    }

    /// Creates an atom term from a raw value (atom id)
    #[inline]
    pub fn make_atom(id: usize) -> Self {
        assert!(id <= constants::MAX_IMMEDIATE2_VALUE);
        Self(constants::make_immediate2(id, Self::FLAG_ATOM))
    }

    /// Executes the destructor for the underlying term, when the
    /// underlying term has a destructor which needs to run, such
    /// as `ProcBin`, which needs to be dropped in order to ensure
    /// that the reference count is decremented properly.
    ///
    /// NOTE: This does not follow move markers, it is assumed that
    /// moved terms are live and not to be released
    #[inline]
    pub fn release(self) {
        // Follow boxed terms and release them
        if self.is_boxed() {
            let boxed_ptr = self.boxed_val();
            let boxed = unsafe { *boxed_ptr };

            // Do not follow moves
            if is_move_marker(boxed) {
                ()
            // Ensure ref-counted binaries are released properly
            } else if boxed.is_procbin() {
                unsafe { ptr::drop_in_place(boxed_ptr as *mut ProcBin) };
            // Ensure ref-counted resources are released properly
            } else if boxed.is_resource_reference_header() {
                unsafe { ptr::drop_in_place(boxed_ptr as *mut resource::Reference) };
            // Ensure we walk tuples and release all their elements
            } else if boxed.is_tuple_header() {
                let tuple = unsafe { &*(boxed_ptr as *mut Tuple) };

                for element in tuple.iter() {
                    element.release();
                }
            }
        // Ensure we walk lists and release all their elements
        } else if self.is_non_empty_list() {
            let cons_ptr = self.list_val();
            let mut cons = unsafe { *cons_ptr };

            loop {
                // Do not follow moves
                if cons.is_move_marker() {
                    break;
                // If we reached the end of the list, we're done
                } else if cons.head.is_nil() {
                    break;
                // Otherwise release the head term
                } else {
                    cons.head.release();
                }

                // This is more of a sanity check, as the head will be nil for EOL
                if cons.tail.is_nil() {
                    break;
                // If the tail is proper, move into the cell it represents
                } else if cons.tail.is_non_empty_list() {
                    let tail_ptr = cons.tail.list_val();
                    cons = unsafe { *tail_ptr };

                    continue;
                // Otherwise if the tail is improper, release it, and we're done
                } else {
                    cons.tail.release();
                }

                break;
            }
        }
    }

    /// Casts the `Term` into its raw `usize` form
    #[inline]
    pub fn as_usize(self) -> usize {
        self.0
    }

    /// Returns `true` if `self` and `other` are equal without converting integers and floats.
    pub fn exactly_eq(self, other: &Term) -> bool {
        let can_be_exactly_equal = match (
            self.to_typed_term().unwrap(),
            other.to_typed_term().unwrap(),
        ) {
            (TypedTerm::SmallInteger(_), TypedTerm::Boxed(_)) => false,
            (TypedTerm::Boxed(_), TypedTerm::SmallInteger(_)) => false,
            (TypedTerm::Boxed(self_unboxed), TypedTerm::Boxed(other_unboxed)) => {
                match (
                    self_unboxed.to_typed_term().unwrap(),
                    other_unboxed.to_typed_term().unwrap(),
                ) {
                    (TypedTerm::BigInteger(_), TypedTerm::Float(_)) => false,
                    (TypedTerm::Float(_), TypedTerm::BigInteger(_)) => false,
                    _ => true,
                }
            }
            _ => true,
        };

        can_be_exactly_equal
            && (self
                .to_typed_term()
                .unwrap()
                .eq(&other.to_typed_term().unwrap()))
    }

    /// Returns `false` if `self` and `other` are equal without converting integers and floats.
    pub fn exactly_ne(self, other: &Term) -> bool {
        !self.exactly_eq(other)
    }

    pub fn is_function(&self) -> bool {
        let tagged = self.0;

        typecheck::is_boxed(tagged) && !constants::is_literal(tagged) && {
            let ptr = constants::boxed_value(tagged);

            unsafe { &*ptr }.is_function_header()
        }
    }

    pub fn is_function_header(&self) -> bool {
        typecheck::is_function(self.0)
    }

    /// Returns true if this term is the none value
    #[inline]
    pub fn is_none(&self) -> bool {
        const_assert!(typecheck::is_none(typecheck::NONE));
        typecheck::is_none(self.0)
    }

    /// Returns true if this term is nil
    #[inline]
    pub fn is_nil(&self) -> bool {
        const_assert!(typecheck::is_nil(typecheck::NIL));
        typecheck::is_nil(self.0)
    }

    /// Returns true if this term is a literal
    #[inline]
    pub fn is_literal(&self) -> bool {
        typecheck::is_literal(self.0)
    }

    /// Returns true if this term is a list
    #[inline]
    pub fn is_list(&self) -> bool {
        self.is_nil() || self.is_non_empty_list()
    }

    pub fn is_non_empty_list(&self) -> bool {
        typecheck::is_list(self.0)
    }

    pub fn is_proper_list(&self) -> bool {
        self.is_nil()
            || (self.is_non_empty_list() && {
                let cons: Boxed<Cons> = (*self).try_into().unwrap();

                cons.is_proper()
            })
    }

    /// Returns true if this a term that the runtime should accept as an argument.
    pub fn is_runtime(&self) -> bool {
        self.is_immediate() || self.is_boxed() || self.is_non_empty_list()
    }

    /// Returns true if this term is an atom
    #[inline]
    pub fn is_atom(&self) -> bool {
        typecheck::is_atom(self.0)
    }

    /// Returns `true` if this term in atom of either `true` or `false`.
    pub fn is_boolean(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Atom(atom) => {
                let name = atom.name();

                (name == "true") || (name == "false")
            }
            _ => false,
        }
    }

    /// Returns true if this term is a number (float or integer)
    #[inline]
    pub fn is_number(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(_) => true,
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) | TypedTerm::Float(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns true if this term is either a small or big integer
    #[inline]
    pub fn is_integer(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(_) => true,
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns true if this term is a small integer (i.e. fits in a usize)
    #[inline]
    pub fn is_smallint(&self) -> bool {
        typecheck::is_smallint(self.0)
    }

    /// Returns true if this term is a boxed big integer (i.e. arbitrarily large).
    #[inline]
    pub fn is_bigint(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns true if this term is a big integer (i.e. arbitrarily large) header that has already
    /// been unboxed.
    #[inline]
    pub fn is_bigint_header(&self) -> bool {
        typecheck::is_bigint(self.0)
    }

    /// Returns true if this term is a boxed float.
    #[inline]
    pub fn is_float(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::Float(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Return true if this term is a float header that has already been unboxed.
    pub fn is_float_header(&self) -> bool {
        typecheck::is_float(self.0)
    }

    /// Returns true if this term is a boxed tuple
    pub fn is_tuple(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => boxed.is_tuple_header(),
            _ => false,
        }
    }

    /// Returns true if this term is a tuple header that has already been unboxed.
    #[inline]
    pub fn is_tuple_header(&self) -> bool {
        typecheck::is_tuple(self.0)
    }

    /// Returns true if this term is a tuple header that has already been unboxed of arity `arity`.
    #[inline]
    pub fn is_tuple_header_with_arity(&self, arity: usize) -> bool {
        if self.is_tuple_header() {
            self.arityval() == arity
        } else {
            false
        }
    }

    /// Returns true if this term is a map
    #[inline]
    pub fn is_map(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => boxed.is_map_header(),
            _ => false,
        }
    }

    /// Returns true if this term is a map header that has already been unboxed.
    pub fn is_map_header(&self) -> bool {
        typecheck::is_map(self.0)
    }

    /// Returns true if this term is a pid
    #[inline]
    pub fn is_pid(&self) -> bool {
        self.is_local_pid() || self.is_external_pid()
    }

    /// Returns true if this term is a pid on the local node
    #[inline]
    pub fn is_local_pid(&self) -> bool {
        typecheck::is_local_pid(self.0)
    }

    /// Returns true if this term is a boxed external pid
    pub fn is_external_pid(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => boxed.is_external_pid_header(),
            _ => false,
        }
    }

    /// Returns true if this term is a pid on some other node
    #[inline]
    pub fn is_external_pid_header(&self) -> bool {
        typecheck::is_remote_pid(self.0)
    }

    /// Returns true if this term is a pid
    #[inline]
    pub fn is_port(&self) -> bool {
        typecheck::is_port(self.0)
    }

    /// Returns true if this term is a port on the local node
    #[inline]
    pub fn is_local_port(&self) -> bool {
        typecheck::is_local_port(self.0)
    }

    /// Returns true if this term is a port on some other node
    #[inline]
    pub fn is_remote_port(&self) -> bool {
        typecheck::is_remote_port(self.0)
    }

    /// Returns true if this term is a boxed reference.
    pub fn is_reference(&self) -> bool {
        let tagged = self.0;

        typecheck::is_boxed(tagged) && !constants::is_literal(tagged) && {
            let ptr = constants::boxed_value(tagged);
            let term = unsafe { &*ptr };

            term.is_local_reference_header()
                || term.is_remote_reference_header()
                || term.is_resource_reference_header()
        }
    }

    /// Returns true if this term is an unboxed reference
    #[inline]
    pub fn is_reference_header(&self) -> bool {
        typecheck::is_reference(self.0)
    }

    /// Returns true if this term is a boxed reference on the local node.
    #[inline]
    pub fn is_local_reference(&self) -> bool {
        let tagged = self.0;

        typecheck::is_boxed(tagged) && !constants::is_literal(tagged) && {
            let ptr = constants::boxed_value(tagged);

            unsafe { &*ptr }.is_local_reference_header()
        }
    }

    /// Returns true if this term is a reference on the local node that has already been unboxed.
    #[inline]
    pub fn is_local_reference_header(&self) -> bool {
        typecheck::is_local_reference(self.0)
    }

    /// Returns true if this term is a reference on some other node
    #[inline]
    pub fn is_remote_reference_header(&self) -> bool {
        typecheck::is_remote_reference(self.0)
    }

    pub fn is_resource_reference(&self) -> bool {
        let tagged = self.0;

        typecheck::is_boxed(tagged) && !constants::is_literal(tagged) && {
            let ptr = constants::boxed_value(tagged);

            unsafe { &*ptr }.is_resource_reference_header()
        }
    }

    pub fn is_resource_reference_header(&self) -> bool {
        typecheck::is_resource_reference(self.0)
    }

    /// Returns true if this term is a reference-counted binary
    #[inline]
    pub fn is_procbin(&self) -> bool {
        typecheck::is_procbin(self.0)
    }

    /// Returns true if this term is a binary on a process heap
    #[inline]
    pub fn is_heapbin(&self) -> bool {
        typecheck::is_heapbin(self.0)
    }

    /// Returns true if this term is a boxed sub-binary.
    pub fn is_subbinary(&self) -> bool {
        typecheck::is_boxed(self.0)
            && unsafe { &*constants::boxed_value(self.0) }.is_subbinary_header()
    }

    /// Returns true if this term is an unboxed sub-binary.
    #[inline]
    pub fn is_subbinary_header(&self) -> bool {
        typecheck::is_subbinary(self.0)
    }

    /// Returns true if this term is a binary match context
    #[inline]
    pub fn is_match_context(&self) -> bool {
        typecheck::is_match_context(self.0)
    }

    /// Returns true if this term is a boxed closure
    pub fn is_closure(&self) -> bool {
        typecheck::is_boxed(self.0)
            && typecheck::is_closure(unsafe { *constants::boxed_value(self.0) }.0)
    }

    /// Returns true if this term is an unboxed closure header
    #[inline]
    pub fn is_closure_header(&self) -> bool {
        typecheck::is_closure(self.0)
    }

    /// Returns true if this term is a catch flag
    #[inline]
    pub fn is_catch(&self) -> bool {
        typecheck::is_catch(self.0)
    }

    /// Returns true if this term is an immediate value
    #[inline]
    pub const fn is_immediate(&self) -> bool {
        typecheck::is_immediate(self.0)
    }

    /// Returns true if this term is a constant value
    ///
    /// NOTE: Currently the meaning of constant in this context is
    /// equivalent to that of immediates, i.e. an immediate is a constant,
    /// and only immediates are constants. Realistically we should be able
    /// to support constants of arbitrary term type, but this is derived
    /// from BEAM for now
    #[inline]
    pub fn is_const(&self) -> bool {
        typecheck::is_const(self.0)
    }

    /// Returns true if this term is a boxed pointer
    #[inline]
    pub fn is_boxed(&self) -> bool {
        typecheck::is_boxed(self.0)
    }

    /// Returns `true` if this term is a boxed pointer and it points to a `HeapBin`, `ProcBin`, or
    /// `SubBinary`, or `MatchContext` with a complete number of bytes.
    pub fn is_binary(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::HeapBinary(_) | TypedTerm::ProcBin(_) => true,
                TypedTerm::SubBinary(subbinary) => subbinary.partial_byte_bit_len() == 0,
                TypedTerm::MatchContext(match_context) => match_context.partial_byte_bit_len() == 0,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns `true` if this term is a boxed pointer and it points to a `HeapBin`, `ProcBin`, or
    /// `SubBinary`, or `MatchContext`
    pub fn is_bitstring(&self) -> bool {
        match self.to_typed_term().unwrap() {
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::HeapBinary(_)
                | TypedTerm::ProcBin(_)
                | TypedTerm::SubBinary(_)
                | TypedTerm::MatchContext(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns true if this term is a header
    #[inline]
    pub fn is_header(&self) -> bool {
        typecheck::is_header(self.0)
    }

    /// Given a boxed term, this function returns a pointer to the boxed value
    ///
    /// NOTE: This is used internally by GC, you should use `to_typed_term` everywhere else
    #[inline]
    pub(crate) fn boxed_val(&self) -> *mut Term {
        assert!(self.is_boxed());
        constants::boxed_value(self.0)
    }

    /// Given a non-empty list term, this function returns a pointer to the underlying `Cons`
    ///
    /// NOTE: This is used internally by GC, you should use `to_typed_term` everywhere else
    #[inline]
    pub(crate) fn list_val(&self) -> *mut Cons {
        assert!(self.is_non_empty_list());
        constants::list_value(self.0)
    }

    /// Given a header term, this function returns the raw arity value,
    /// i.e. all flags are stripped, leaving the remaining bits untouched
    ///
    /// NOTE: This function will panic if called on anything but a header term
    #[inline]
    pub fn arityval(&self) -> usize {
        assert!(self.is_header());
        constants::header_value(self.0)
    }

    /// Returns `true` if the underlying direct type of the term has no arity, so any ptr math can
    /// increment by `1`.
    pub fn has_no_arity(&self) -> bool {
        self.is_immediate() || self.is_boxed() || self.is_non_empty_list()
    }

    /// Given a pointer to a generic term, converts it to its typed representation
    pub fn to_typed_term(&self) -> Result<TypedTerm, InvalidTermError> {
        let val = self.0;
        match constants::primary_tag(val) {
            Self::FLAG_HEADER => unsafe { Self::header_to_typed_term(self, val) },
            Self::FLAG_LIST => {
                let ptr = constants::list_value(val);
                if constants::is_literal(val) {
                    Ok(TypedTerm::List(unsafe { Boxed::from_raw_literal(ptr) }))
                } else {
                    Ok(TypedTerm::List(unsafe { Boxed::from_raw(ptr) }))
                }
            }
            Self::FLAG_BOXED => {
                let ptr = constants::boxed_value(val);
                if constants::is_literal(val) {
                    Ok(TypedTerm::Literal(unsafe { *ptr }))
                } else {
                    Ok(TypedTerm::Boxed(unsafe { Boxed::from_raw(ptr) }))
                }
            }
            Self::FLAG_IMMEDIATE => match constants::immediate1_tag(val) {
                Self::FLAG_PID => Ok(TypedTerm::Pid(unsafe {
                    Pid::from_raw(constants::immediate1_value(val))
                })),
                Self::FLAG_PORT => Ok(TypedTerm::Port(unsafe {
                    Port::from_raw(constants::immediate1_value(val))
                })),
                Self::FLAG_IMMEDIATE2 => match constants::immediate2_tag(val) {
                    Self::FLAG_ATOM => Ok(TypedTerm::Atom(unsafe {
                        Atom::from_id(constants::immediate2_value(val))
                    })),
                    Self::FLAG_CATCH => Ok(TypedTerm::Catch),
                    Self::FLAG_UNUSED_1 => Err(InvalidTermError::InvalidTag),
                    Self::FLAG_NIL => Ok(TypedTerm::Nil),
                    _ => Err(InvalidTermError::InvalidTag),
                },
                Self::FLAG_SMALL_INTEGER => {
                    let i = constants::smallint_value(val);
                    let small = unsafe { SmallInteger::new_unchecked(i) };
                    Ok(TypedTerm::SmallInteger(small))
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    #[inline]
    unsafe fn header_to_typed_term(ptr: &Term, val: usize) -> Result<TypedTerm, InvalidTermError> {
        let ptr = ptr as *const _ as *mut Term;
        let ty = match constants::header_tag(val) {
            Self::FLAG_TUPLE => TypedTerm::Tuple(Boxed::from_raw(ptr as *mut Tuple)),
            Self::FLAG_NONE => {
                if constants::header_value(val) == constants::NONE_VAL {
                    TypedTerm::None
                } else {
                    // Garbage value
                    return Err(InvalidTermError::InvalidTag);
                }
            }
            Self::FLAG_POS_BIG_INTEGER => {
                TypedTerm::BigInteger(Boxed::from_raw(ptr as *mut BigInteger))
            }
            Self::FLAG_NEG_BIG_INTEGER => {
                TypedTerm::BigInteger(Boxed::from_raw(ptr as *mut BigInteger))
            }
            Self::FLAG_REFERENCE => TypedTerm::Reference(Boxed::from_raw(ptr as *mut Reference)), /* RefThing in erl_term.h */
            Self::FLAG_CLOSURE => TypedTerm::Closure(Boxed::from_raw(ptr as *mut Closure)), /* ErlFunThing in erl_fun.h */
            Self::FLAG_FLOAT => TypedTerm::Float(Float::from_raw(ptr as *mut Float)),
            Self::FLAG_RESOURCE_REFERENCE => TypedTerm::ResourceReference(
                resource::Reference::from_raw(ptr as *mut resource::Reference),
            ),
            Self::FLAG_PROCBIN => TypedTerm::ProcBin(ProcBin::from_raw(ptr as *mut ProcBin)),
            Self::FLAG_HEAPBIN => TypedTerm::HeapBinary(Boxed::from_raw(ptr as *mut HeapBin)),
            Self::FLAG_SUBBINARY => {
                TypedTerm::SubBinary(SubBinary::from_raw(ptr as *mut SubBinary))
            }
            Self::FLAG_MATCH_CTX => {
                TypedTerm::MatchContext(MatchContext::from_raw(ptr as *mut MatchContext))
            }
            Self::FLAG_EXTERN_PID => {
                TypedTerm::ExternalPid(Boxed::from_raw(ptr as *mut ExternalPid))
            }
            Self::FLAG_EXTERN_PORT => {
                TypedTerm::ExternalPort(Boxed::from_raw(ptr as *mut ExternalPort))
            }
            Self::FLAG_EXTERN_REF => {
                TypedTerm::ExternalReference(Boxed::from_raw(ptr as *mut ExternalReference))
            }
            Self::FLAG_MAP => TypedTerm::Map(Boxed::from_raw(ptr as *mut Map)),
            _ => unreachable!(),
        };
        Ok(ty)
    }
}
// Needed so that `Boxed` can be `Boxed<Term>` when `to_typed_term` returns a
// `TypedTerm::Boxed(Boxed<Term>)`.
unsafe impl AsTerm for Term {
    unsafe fn as_term(&self) -> Term {
        *self
    }
}

impl Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_boxed() {
            let ptr = self.boxed_val();
            let boxed = unsafe { *ptr };
            if is_move_marker(boxed) {
                let to = boxed.boxed_val();
                write!(f, "Term(Moved({:?} => {:?}))", ptr, to)
            } else {
                let ptr = self.boxed_val();
                let unboxed = unsafe { &*ptr };
                write!(f, "Term(Boxed({:?} => {:?}))", ptr, unboxed)
            }
        } else if self.is_non_empty_list() {
            let ptr = self.list_val();
            let cons = unsafe { &*ptr };
            if cons.is_move_marker() {
                let to = unsafe { &*cons.tail.list_val() };
                write!(f, "Term(Moved({:?} => {:?}))", ptr, to)
            } else {
                write!(f, "Term({:?})", cons)
            }
        } else if self.is_immediate() {
            if self.is_atom() {
                let id = constants::immediate2_value(self.0);
                let atom = unsafe { Atom::from_id(id) };
                write!(f, "Term({:?}, id: {})", atom, id)
            } else if self.is_smallint() {
                write!(f, "Term({})", unsafe {
                    SmallInteger::new_unchecked(constants::smallint_value(self.0))
                })
            } else if self.is_pid() {
                write!(f, "Term({:?})", unsafe {
                    Pid::from_raw(constants::immediate1_value(self.0))
                })
            } else if self.is_port() {
                write!(f, "Term({:?})", unsafe {
                    Port::from_raw(constants::immediate1_value(self.0))
                })
            } else if self.is_nil() {
                write!(f, "Term(Nil)")
            } else if self.is_catch() {
                write!(f, "Term(Catch)")
            } else {
                unreachable!()
            }
        } else if self.is_header() {
            let ptr = self as *const _;
            unsafe {
                if self.is_tuple_header() {
                    write!(f, "Term({:?})", &*(ptr as *const Tuple))
                } else if self.is_none() {
                    write!(f, "Term(None)")
                } else if self.is_bigint_header() {
                    write!(f, "Term({})", &*(ptr as *const BigInteger))
                } else if self.is_local_reference_header() {
                    write!(f, "Term({:?})", &*(ptr as *mut Reference))
                } else if self.is_closure_header() {
                    write!(f, "Term(Closure({:?}))", &*(ptr as *const Closure))
                } else if self.is_float_header() {
                    write!(f, "Term({})", &*(ptr as *mut Float))
                } else if self.is_procbin() {
                    let bin = &*(ptr as *const ProcBin);
                    if bin.is_raw() {
                        write!(f, "Term(ProcBin({:?}))", bin.as_bytes())
                    } else {
                        write!(f, "Term(ProcBin({}))", bin.as_str())
                    }
                } else if self.is_heapbin() {
                    let bin = &*(ptr as *const HeapBin);
                    if bin.is_raw() {
                        write!(f, "Term(HeapBin({:?}))", bin.as_bytes())
                    } else {
                        write!(f, "Term(HeapBin({}))", bin.as_str())
                    }
                } else if self.is_subbinary_header() {
                    let bin = &*(ptr as *const SubBinary);
                    write!(f, "Term({:?})", bin)
                } else if self.is_match_context() {
                    let bin = &*(ptr as *const MatchContext);
                    write!(f, "Term(MatchCtx({:?}))", bin)
                } else if self.is_external_pid_header() {
                    let val = &*(ptr as *const ExternalPid);
                    write!(f, "Term({:?})", val)
                } else if self.is_remote_port() {
                    let val = &*(ptr as *const ExternalPort);
                    write!(f, "Term({:?})", val)
                } else if self.is_remote_reference_header() {
                    let val = &*(ptr as *const ExternalReference);
                    write!(f, "Term({:?})", val)
                } else if self.is_map_header() {
                    let val = &*(ptr as *const Map);
                    write!(f, "Term({:?})", val)
                } else if self.is_resource_reference_header() {
                    let val = &*(ptr as *const resource::Reference);
                    write!(f, "Term({:?})", val)
                } else {
                    write!(f, "Term(UnknownHeader({:#x}))", self.0)
                }
            }
        } else {
            write!(f, "Term(UnknownPrimary({:?}))", self.0)
        }
    }
}

impl Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_typed_term().unwrap())
    }
}

impl From<bool> for Term {
    fn from(b: bool) -> Term {
        atom_unchecked(&b.to_string())
    }
}
impl From<u8> for Term {
    fn from(byte: u8) -> Term {
        let small_integer: SmallInteger = byte.into();

        unsafe { small_integer.as_term() }
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_typed_term().unwrap().hash(state)
    }
}

impl PartialEq<Term> for Term {
    fn eq(&self, other: &Term) -> bool {
        match (self.to_typed_term(), other.to_typed_term()) {
            (Ok(ref self_typed_term), Ok(ref other_typed_term)) => {
                self_typed_term.eq(other_typed_term)
            }
            (Err(_), Err(_)) => true,
            _ => false,
        }
    }
}
impl PartialOrd<Term> for Term {
    fn partial_cmp(&self, other: &Term) -> Option<cmp::Ordering> {
        if let Ok(ref lhs) = self.to_typed_term() {
            if let Ok(ref rhs) = other.to_typed_term() {
                return lhs.partial_cmp(rhs);
            }
        }
        None
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Term) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl CloneToProcess for Term {
    fn clone_to_process(&self, process: &ProcessControlBlock) -> Term {
        if self.is_immediate() {
            *self
        } else if self.is_boxed() || self.is_non_empty_list() {
            let tt = self.to_typed_term().unwrap();
            tt.clone_to_process(process)
        } else {
            panic!("clone_to_process called on invalid term type: {:?}", self);
        }
    }

    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        debug_assert!(self.is_runtime());
        if self.is_immediate() {
            Ok(*self)
        } else if self.is_boxed() || self.is_non_empty_list() {
            let tt = self.to_typed_term().unwrap();
            tt.clone_to_heap(heap)
        } else {
            panic!("clone_to_heap called on invalid term type: {:?}", self);
        }
    }

    fn size_in_words(&self) -> usize {
        if self.is_immediate() {
            return 1;
        } else if self.is_boxed() || self.is_non_empty_list() {
            let tt = self.to_typed_term().unwrap();
            tt.size_in_words()
        } else {
            assert!(self.is_header());
            let arityval = self.arityval();
            arityval + 1
        }
    }
}

unsafe impl Send for Term {}

impl TryInto<bool> for Term {
    type Error = BoolError;

    fn try_into(self) -> Result<bool, Self::Error> {
        self.to_typed_term().unwrap().try_into()
    }
}

impl TryInto<char> for Term {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<char, Self::Error> {
        let self_u32: u32 = self
            .try_into()
            .map_err(|_| TryIntoIntegerError::OutOfRange)?;

        match core::char::from_u32(self_u32) {
            Some(c) => Ok(c),
            None => Err(TryIntoIntegerError::OutOfRange),
        }
    }
}

impl TryInto<f64> for Term {
    type Error = TypeError;

    fn try_into(self) -> Result<f64, Self::Error> {
        self.to_typed_term().unwrap().try_into()
    }
}

impl TryInto<isize> for Term {
    type Error = TypeError;

    fn try_into(self) -> Result<isize, Self::Error> {
        self.to_typed_term().unwrap().try_into()
    }
}

impl TryInto<u32> for Term {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<u32, Self::Error> {
        let u: u64 = self.try_into()?;

        u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
    }
}

impl TryInto<u64> for Term {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<u64, Self::Error> {
        match self.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(small_integer) => small_integer
                .try_into()
                .map_err(|_| TryIntoIntegerError::OutOfRange),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(big_integer) => big_integer.try_into(),
                _ => Err(TryIntoIntegerError::Type),
            },
            _ => Err(TryIntoIntegerError::Type),
        }
    }
}

impl TryInto<usize> for Term {
    type Error = TryIntoIntegerError;

    fn try_into(self) -> Result<usize, Self::Error> {
        let u: u64 = self.try_into()?;

        u.try_into().map_err(|_| TryIntoIntegerError::OutOfRange)
    }
}

impl TryInto<BigInt> for Term {
    type Error = TypeError;

    fn try_into(self) -> Result<BigInt, Self::Error> {
        let option_big_int = match self.to_typed_term().unwrap() {
            TypedTerm::SmallInteger(small_integer) => Some(small_integer.into()),
            TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                TypedTerm::BigInteger(big_integer) => {
                    let big_int: BigInt = big_integer.clone().into();

                    Some(big_int.clone())
                }
                _ => None,
            },
            _ => None,
        };

        match option_big_int {
            Some(big_int) => Ok(big_int),
            None => Err(TypeError),
        }
    }
}

impl TryInto<String> for Term {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<String, Self::Error> {
        self.to_typed_term().unwrap().try_into()
    }
}

impl TryInto<Vec<u8>> for Term {
    type Error = runtime::Exception;

    fn try_into(self) -> Result<Vec<u8>, Self::Error> {
        self.to_typed_term().unwrap().try_into()
    }
}

pub enum BoolError {
    Type,
    NotABooleanName,
}

#[derive(Debug)]
pub struct TypeError;

#[derive(Debug)]
pub enum TryIntoIntegerError {
    Type,
    OutOfRange,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxed_term_invariants() {
        assert!(typecheck::is_boxed(0 | constants::FLAG_BOXED));
        assert!(typecheck::is_boxed(
            constants::MAX_ALIGNED_ADDR | constants::FLAG_BOXED
        ));
        assert!(!typecheck::is_list(
            constants::MAX_ALIGNED_ADDR | constants::FLAG_BOXED
        ));
        assert!(!typecheck::is_header(
            constants::MAX_ALIGNED_ADDR | constants::FLAG_BOXED
        ));
        assert_eq!(
            constants::boxed_value(constants::MAX_ALIGNED_ADDR | constants::FLAG_BOXED) as usize,
            constants::MAX_ALIGNED_ADDR
        );
    }

    #[test]
    fn literal_term_invariants() {
        assert!(typecheck::is_literal(
            0 | constants::FLAG_BOXED | constants::FLAG_LITERAL
        ));
        assert!(typecheck::is_literal(
            constants::MAX_ALIGNED_ADDR | constants::FLAG_BOXED | constants::FLAG_LITERAL
        ));
        assert!(typecheck::is_literal(
            0 | constants::FLAG_LIST | constants::FLAG_LITERAL
        ));
        assert!(typecheck::is_literal(
            constants::MAX_ALIGNED_ADDR | constants::FLAG_LIST | constants::FLAG_LITERAL
        ));
        assert!(Term::make_boxed_literal(constants::MAX_ALIGNED_ADDR as *mut Term).is_literal());
    }

    #[test]
    fn atom_term_invariants() {
        let a = constants::make_immediate2(1, constants::FLAG_ATOM);
        let b = constants::make_immediate2(constants::MAX_IMMEDIATE2_VALUE, constants::FLAG_ATOM);
        assert!(typecheck::is_atom(a));
        assert!(typecheck::is_atom(b));
        assert_eq!(constants::immediate1_tag(a), constants::FLAG_ATOM);
        assert_eq!(constants::immediate1_tag(b), constants::FLAG_ATOM);

        let c = Term::make_atom(1);
        assert!(c.is_atom());
    }

    #[test]
    fn smallint_term_invariants() {
        let min = constants::make_smallint(constants::MIN_SMALLINT_VALUE);
        let negative_one = constants::make_smallint(-1);
        let zero = constants::make_smallint(0);
        let one = constants::make_smallint(1);
        let max = constants::make_smallint(constants::MAX_SMALLINT_VALUE);

        assert!(constants::MIN_SMALLINT_VALUE < -1);
        assert!(typecheck::is_smallint(min));
        assert!(typecheck::is_integer(min));
        assert!(typecheck::is_number(min));
        assert_eq!(
            constants::smallint_value(min),
            constants::MIN_SMALLINT_VALUE
        );

        assert!(typecheck::is_smallint(negative_one));
        assert!(typecheck::is_integer(negative_one));
        assert!(typecheck::is_number(negative_one));
        assert_eq!(constants::smallint_value(negative_one), -1);

        assert!(typecheck::is_smallint(zero));
        assert!(typecheck::is_integer(zero));
        assert!(typecheck::is_number(zero));
        assert_eq!(constants::smallint_value(zero), 0);

        assert!(typecheck::is_smallint(one));
        assert!(typecheck::is_integer(one));
        assert!(typecheck::is_number(one));
        assert_eq!(constants::smallint_value(one), 1);

        assert!(1 < constants::MAX_SMALLINT_VALUE);
        assert!(typecheck::is_smallint(max));
        assert!(typecheck::is_integer(max));
        assert!(typecheck::is_number(max));
        assert_eq!(
            constants::smallint_value(max),
            constants::MAX_SMALLINT_VALUE
        );
    }

    #[test]
    fn list_term_invariants() {
        let a = constants::make_list(ptr::null());
        let b = constants::make_list(constants::MAX_ALIGNED_ADDR as *const Cons);
        assert!(typecheck::is_list(a as usize));
        assert!(typecheck::is_list(b as usize));
        assert!(!typecheck::is_boxed(a));
        assert!(!typecheck::is_boxed(b));
        assert!(!typecheck::is_header(a));
        assert!(!typecheck::is_header(b));
        assert_eq!(
            constants::list_value(a) as usize,
            ptr::null::<Term>() as usize
        );
        assert_eq!(
            constants::list_value(b) as usize,
            constants::MAX_ALIGNED_ADDR
        );

        let c = Term::make_list(ptr::null());
        assert!(c.is_list());
    }

    #[test]
    fn is_number_invariants() {
        assert!(typecheck::is_number(constants::FLAG_SMALL_INTEGER));
        assert!(typecheck::is_number(constants::FLAG_POS_BIG_INTEGER));
        assert!(typecheck::is_number(constants::FLAG_NEG_BIG_INTEGER));
        assert!(typecheck::is_float(constants::FLAG_FLOAT));
        assert!(!typecheck::is_number(constants::FLAG_PID));
        assert!(!typecheck::is_number(constants::FLAG_PORT));
        assert!(!typecheck::is_number(constants::FLAG_ATOM));
    }

    #[test]
    fn is_integer_invariants() {
        assert!(typecheck::is_integer(constants::FLAG_SMALL_INTEGER));
        assert!(typecheck::is_integer(constants::FLAG_POS_BIG_INTEGER));
        assert!(typecheck::is_integer(constants::FLAG_NEG_BIG_INTEGER));
        assert!(!typecheck::is_integer(constants::FLAG_FLOAT));
    }

    #[test]
    fn is_smallint_invariants() {
        assert!(typecheck::is_smallint(constants::FLAG_SMALL_INTEGER));
        assert!(!typecheck::is_smallint(constants::FLAG_POS_BIG_INTEGER));
        assert!(!typecheck::is_smallint(constants::FLAG_NEG_BIG_INTEGER));
        assert!(!typecheck::is_smallint(constants::FLAG_FLOAT));
    }

    #[test]
    fn is_bigint_invariants() {
        assert!(!typecheck::is_bigint(constants::FLAG_SMALL_INTEGER));
        assert!(typecheck::is_bigint(constants::FLAG_POS_BIG_INTEGER));
        assert!(typecheck::is_bigint(constants::FLAG_NEG_BIG_INTEGER));
        assert!(!typecheck::is_bigint(constants::FLAG_FLOAT));
    }

    #[test]
    fn is_float_invariants() {
        assert!(!typecheck::is_float(constants::FLAG_SMALL_INTEGER));
        assert!(!typecheck::is_float(constants::FLAG_POS_BIG_INTEGER));
        assert!(!typecheck::is_float(constants::FLAG_NEG_BIG_INTEGER));
        assert!(typecheck::is_float(constants::FLAG_FLOAT));
    }

    #[test]
    fn is_tuple_invariants() {
        assert!(typecheck::is_header(constants::make_header(
            2,
            constants::FLAG_TUPLE
        )));
        assert!(typecheck::is_tuple(constants::make_header(
            2,
            constants::FLAG_TUPLE
        )));
    }

    #[test]
    fn is_map_invariants() {
        assert!(typecheck::is_header(constants::FLAG_MAP));
        assert!(typecheck::is_map(constants::FLAG_MAP));
    }

    #[test]
    fn is_pid_invariants() {
        assert!(typecheck::is_immediate(constants::FLAG_PID));
        assert!(typecheck::is_pid(constants::FLAG_PID));
        assert!(typecheck::is_header(constants::FLAG_EXTERN_PID));
        assert!(typecheck::is_pid(constants::FLAG_EXTERN_PID));
    }

    #[test]
    fn is_local_pid_invariants() {
        assert!(typecheck::is_local_pid(constants::FLAG_PID));
        assert!(!typecheck::is_local_pid(constants::FLAG_EXTERN_PID));
    }

    #[test]
    fn is_remote_pid_invariants() {
        assert!(!typecheck::is_remote_pid(constants::FLAG_PID));
        assert!(typecheck::is_remote_pid(constants::FLAG_EXTERN_PID));
    }

    #[test]
    fn is_port_invariants() {
        assert!(typecheck::is_port(constants::FLAG_PORT));
        assert!(typecheck::is_port(constants::FLAG_EXTERN_PORT));
    }

    #[test]
    fn is_local_port_invariants() {
        assert!(typecheck::is_immediate(constants::FLAG_PORT));
        assert!(typecheck::is_local_port(constants::FLAG_PORT));
        assert!(!typecheck::is_local_port(constants::FLAG_EXTERN_PORT));
    }

    #[test]
    fn is_remote_port_invariants() {
        assert!(!typecheck::is_remote_port(constants::FLAG_PORT));
        assert!(typecheck::is_header(constants::FLAG_EXTERN_PORT));
        assert!(typecheck::is_remote_port(constants::FLAG_EXTERN_PORT));
    }

    #[test]
    fn is_reference_invariants() {
        assert!(typecheck::is_reference(constants::FLAG_REFERENCE));
        assert!(typecheck::is_reference(constants::FLAG_EXTERN_REF));
    }

    #[test]
    fn is_local_reference_invariants() {
        assert!(typecheck::is_header(constants::FLAG_REFERENCE));
        assert!(typecheck::is_local_reference(constants::FLAG_REFERENCE));
        assert!(!typecheck::is_local_reference(constants::FLAG_EXTERN_REF));
    }

    #[test]
    fn is_remote_reference_invariants() {
        assert!(!typecheck::is_remote_reference(constants::FLAG_REFERENCE));
        assert!(typecheck::is_header(constants::FLAG_EXTERN_REF));
        assert!(typecheck::is_remote_reference(constants::FLAG_EXTERN_REF));
    }

    #[test]
    fn is_binary_invariants() {
        assert!(typecheck::is_bitstring(constants::FLAG_PROCBIN));
        assert!(typecheck::is_bitstring(constants::FLAG_HEAPBIN));
        assert!(typecheck::is_bitstring(constants::FLAG_SUBBINARY));
        assert!(typecheck::is_bitstring(constants::FLAG_MATCH_CTX));
    }

    #[test]
    fn is_procbin_invariants() {
        assert!(typecheck::is_header(constants::FLAG_PROCBIN));
        assert!(typecheck::is_procbin(constants::FLAG_PROCBIN));
        assert!(!typecheck::is_procbin(constants::FLAG_HEAPBIN));
        assert!(!typecheck::is_procbin(constants::FLAG_SUBBINARY));
        assert!(!typecheck::is_procbin(constants::FLAG_MATCH_CTX));
    }

    #[test]
    fn is_heapbin_invariants() {
        assert!(typecheck::is_header(constants::FLAG_HEAPBIN));
        assert!(!typecheck::is_heapbin(constants::FLAG_PROCBIN));
        assert!(typecheck::is_heapbin(constants::FLAG_HEAPBIN));
        assert!(!typecheck::is_heapbin(constants::FLAG_SUBBINARY));
        assert!(!typecheck::is_heapbin(constants::FLAG_MATCH_CTX));
    }

    #[test]
    fn is_subbinary_invariants() {
        assert!(typecheck::is_header(constants::FLAG_SUBBINARY));
        assert!(!typecheck::is_subbinary(constants::FLAG_PROCBIN));
        assert!(!typecheck::is_subbinary(constants::FLAG_HEAPBIN));
        assert!(typecheck::is_subbinary(constants::FLAG_SUBBINARY));
        assert!(!typecheck::is_subbinary(constants::FLAG_MATCH_CTX));
    }

    #[test]
    fn is_match_context_invariants() {
        assert!(typecheck::is_header(constants::FLAG_MATCH_CTX));
        assert!(!typecheck::is_match_context(constants::FLAG_PROCBIN));
        assert!(!typecheck::is_match_context(constants::FLAG_HEAPBIN));
        assert!(!typecheck::is_match_context(constants::FLAG_SUBBINARY));
        assert!(typecheck::is_match_context(constants::FLAG_MATCH_CTX));
    }

    #[test]
    fn is_closure_invariants() {
        assert!(typecheck::is_header(constants::FLAG_CLOSURE));
        assert!(typecheck::is_closure(constants::FLAG_CLOSURE));
    }

    #[test]
    fn is_catch_invariants() {
        assert!(typecheck::is_immediate(constants::FLAG_CATCH));
        assert!(typecheck::is_catch(constants::FLAG_CATCH));
    }
}
