pub mod arch_32;
pub mod arch_64;
pub mod arch_64_nanboxed;

pub use self::arch_32::Encoding32;
pub use self::arch_64::Encoding64;
pub use self::arch_64_nanboxed::Encoding64Nanboxed;

use core::fmt::{Binary, Debug};
use core::hash::Hash;
use core::mem;

use crate::Tag;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct MaskInfo {
    pub shift: i32,
    pub mask: u64,
    pub max_allowed_value: u64,
}

pub trait Word:
    Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Hash
    + Debug
    + Binary
    + TryInto<usize>
    + TryInto<u32>
    + TryInto<u64>
    + TryFrom<usize>
    + TryFrom<u64>
{
    fn as_usize(&self) -> usize;
}

impl Word for u32 {
    fn as_usize(&self) -> usize {
        (*self) as usize
    }
}
impl Word for u64 {
    fn as_usize(&self) -> usize {
        (*self) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingType {
    /// Use the default encoding based on target pointer width
    Default,
    /// Use a 32-bit encoding
    Encoding32,
    /// Use a 64-bit encoding
    Encoding64,
    /// An alternative 64-bit encoding, based on NaN-boxing
    Encoding64Nanboxed,
}
impl EncodingType {
    #[inline(always)]
    pub fn is_nanboxed(self) -> bool {
        self == Self::Encoding64Nanboxed
    }
}

pub trait Encoding {
    // The concrete type of the encoded term
    type Type: Word;
    // The concrete type of a signed fixed-width integer value (i.e. i64)
    type SignedType;

    // The valid range of integer values that can fit in a term with primary tag
    const MAX_IMMEDIATE_VALUE: Self::Type;
    // The largest atom id supported
    const MAX_ATOM_ID: Self::Type;
    // The minimum value of a fixed-width integer value
    const MIN_SMALLINT_VALUE: Self::SignedType;
    // The maximum value of a fixed-width integer value
    const MAX_SMALLINT_VALUE: Self::SignedType;

    // The default "none" value as a constant
    const NONE: Self::Type;
    // The constant value for 'nil' (with tag)
    const NIL: Self::Type;
    // The constant value for the atom 'true' (without tag)
    const TRUE: Self::Type;
    // The constant value for the atom 'false' (without tag)
    const FALSE: Self::Type;

    fn type_of(value: Self::Type) -> Tag<Self::Type>;

    fn immediate_mask_info() -> MaskInfo;
    fn header_mask_info() -> MaskInfo;

    fn encode_immediate(value: Self::Type, tag: Self::Type) -> Self::Type;

    fn encode_immediate_with_tag(value: Self::Type, tag: Tag<Self::Type>) -> Self::Type;

    fn immediate_tag(tag: Tag<Self::Type>) -> Self::Type;

    fn header_tag(tag: Tag<Self::Type>) -> Self::Type;

    fn encode_list<T: ?Sized>(value: *const T) -> Self::Type;

    fn encode_box<T: ?Sized>(value: *const T) -> Self::Type;

    fn encode_literal<T: ?Sized>(value: *const T) -> Self::Type;

    fn encode_header(value: Self::Type, tag: Self::Type) -> Self::Type;

    fn encode_header_with_tag(value: Self::Type, tag: Tag<Self::Type>) -> Self::Type;

    unsafe fn decode_box<T>(value: Self::Type) -> *mut T;

    unsafe fn decode_list<T>(value: Self::Type) -> *mut T;

    fn decode_smallint(value: Self::Type) -> Self::SignedType;

    fn decode_immediate(value: Self::Type) -> Self::Type;

    #[inline(always)]
    fn decode_atom(value: Self::Type) -> Self::Type {
        Self::decode_immediate(value)
    }

    #[inline(always)]
    fn decode_pid(value: Self::Type) -> Self::Type {
        Self::decode_immediate(value)
    }

    #[inline(always)]
    fn decode_port(value: Self::Type) -> Self::Type {
        Self::decode_immediate(value)
    }

    fn decode_header_value(value: Self::Type) -> Self::Type;

    /// Returns `true` if the encoded value represents `NONE`
    fn is_none(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value represents a pointer to a term
    fn is_boxed(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is the header of a non-immediate term
    fn is_header(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is an immediate value
    fn is_immediate(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value represents a pointer to a literal value
    fn is_literal(value: Self::Type) -> bool;

    /// Returns `true` if the encoded value represents the empty list
    fn is_nil(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value represents a nil or `Cons` value (empty or non-empty
    /// list)
    fn is_list(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value represents a `Cons` value (non-empty list)
    #[inline(always)]
    fn is_non_empty_list(value: Self::Type) -> bool {
        Self::is_list(value) && !Self::is_nil(value)
    }
    /// Returns `true` if the encoded value is an atom
    fn is_atom(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a boolean
    fn is_boolean(value: Self::Type) -> bool {
        if !Self::is_atom(value) {
            return false;
        }
        let atom = Self::decode_atom(value);
        atom == Self::TRUE || atom == Self::FALSE
    }
    /// Returns `true` if the encoded value is a fixed-width integer value
    fn is_smallint(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is the header of a arbitrary-width integer value
    fn is_bigint(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a arbitrary-width integer value
    fn is_boxed_bigint(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_bigint(header)
    }
    /// Returns `true` if the encoded value is a float
    ///
    /// NOTE: This function returns true if either the term is an immediate float,
    /// or if it is the header of a packed float. It does not unwrap boxed values.
    fn is_float(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Float`
    fn is_boxed_float(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_float(header)
    }
    /// Returns `true` if the encoded value is the header of a `Tuple`
    fn is_tuple(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Tuple`
    fn is_boxed_tuple(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_tuple(header)
    }
    /// Returns `true` if the encoded value is the header of a `Map`
    fn is_map(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Map`
    fn is_boxed_map(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_map(header)
    }
    /// Returns `true` if the encoded value is a `Pid`
    fn is_local_pid(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is the header of an `ExternalPid`
    fn is_remote_pid(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to an `ExternalPid`
    fn is_boxed_remote_pid(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_remote_pid(header)
    }
    /// Returns `true` if the encoded value is a `Port`
    fn is_local_port(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is the header of an `ExternalPort`
    fn is_remote_port(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to an `ExternalPort`
    fn is_boxed_remote_port(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_remote_port(header)
    }
    /// Returns `true` if the encoded value is the header of a `Reference`
    fn is_local_reference(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Reference`
    fn is_boxed_local_reference(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_local_reference(header)
    }
    /// Returns `true` if the encoded value is the header of a `ExternalReference`
    fn is_remote_reference(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `ExternalReference`
    fn is_boxed_remote_reference(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_remote_reference(header)
    }
    /// Returns `true` if the encoded value is the header of a `Resource`
    fn is_resource_reference(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Resource`
    fn is_boxed_resource_reference(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_resource_reference(header)
    }
    /// Returns `true` if the encoded value is the header of a `ProcBin`
    fn is_procbin(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `ProcBin`
    fn is_boxed_procbin(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_procbin(header)
    }
    /// Returns `true` if the encoded value is the header of a `HeapBin`
    fn is_heapbin(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `HeapBin`
    fn is_boxed_heapbin(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_heapbin(header)
    }
    /// Returns `true` if the encoded value is the header of a `SubBinary`
    fn is_subbinary(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `SubBinary`
    fn is_boxed_subbinary(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_subbinary(header)
    }
    /// Returns `true` if the encoded value is the header of a `MatchContext`
    fn is_match_context(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `MatchContext`
    fn is_boxed_match_context(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_match_context(header)
    }
    /// Returns `true` if the encoded value is the header of a `Closure`
    fn is_function(value: Self::Type) -> bool;
    /// Returns `true` if the encoded value is a pointer to a `Tuple`
    fn is_boxed_function(value: Self::Type) -> bool {
        if !Self::is_boxed(value) {
            return false;
        }
        let header = unsafe { *(Self::decode_box(value)) };
        Self::is_function(header)
    }

    /// Returns true if this is a term that is valid for use as an argument
    /// in the runtime, as a key in a datstructure, or other position in which
    /// an immediate or a reference is required or desirable
    fn is_valid(value: Self::Type) -> bool {
        !Self::is_none(value) && !Self::is_header(value)
    }

    /// Returns the size in bytes of the term in memory
    fn sizeof(value: Self::Type) -> usize {
        mem::size_of::<Self::Type>() * (Self::arity(value) + 1)
    }

    /// Returns the arity of this term, which reflects the number of words of data
    /// following this term in memory.
    ///
    /// Returns zero for immediates/pointers
    fn arity(value: Self::Type) -> usize {
        if Self::is_header(value) {
            Self::Type::as_usize(&Self::decode_header_value(value))
        } else {
            // All other term types are required to be immediate/word-sized,
            // and as such, have no arity
            0
        }
    }
}
