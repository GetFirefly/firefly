mod heap;
mod match_context;
mod process;
mod literal;
mod sub;
mod iter;
mod compare;
mod aligned_binary;
mod maybe_aligned_maybe_binary;
mod primitives;

use core::str::Utf8Error;

use thiserror::Error;

use crate::erts::string::Encoding;
use crate::erts::exception::Alloc;

use super::prelude::Boxed;

// This module provides a limited set of exported types/traits for convenience
pub mod prelude {
    // Expose the iterator traits for bytes/bits
    pub use super::iter::{ByteIterator, BitIterator};
    // Expose the concrete iterator types for use within the `binary` module only
    pub(super) use super::iter::{FullByteIter, BitsIter, PartialByteBitIter};
    // Expose the various binary/bitstring traits
    pub use super::{Bitstring, Binary, IndexByte, MaybePartialByte};
    pub use super::aligned_binary::AlignedBinary;
    pub use super::maybe_aligned_maybe_binary::MaybeAlignedMaybeBinary;
    // Expose the type for binary flags
    pub use super::BinaryFlags;
    // Expose the concrete binary types
    pub use super::heap::HeapBin;
    pub use super::process::ProcBin;
    pub use super::literal::BinaryLiteral;
    pub use super::match_context::MatchContext;
    pub use super::sub::SubBinary;
    // Expose the error types
    pub use super::{BytesFromBinaryError, StrFromBinaryError};

    // Expose the low-level binary helpers
    pub use super::primitives::CopyDirection;
    pub use super::primitives::{copy_bits, copy_binary_to_buffer};

    // Expose the low-level binary helpers that are restricted to the `binary` module only
    pub(super) use super::primitives::{num_bytes, bit_offset, byte_offset};
}

/// This trait provides common behavior for all types which represent
/// binary data, either as a collection of bytes, or a collection of bits.
///
/// Bitstrings are strictly a superset of binaries, as binaries are simply
/// bitstrings which have a number of bits divisible by 8
pub trait Bitstring {
    /// The total number of full bytes, not including any final partial byte.
    fn full_byte_len(&self) -> usize;

    /// Returns a raw pointer to the data underlying this binary
    ///
    /// # Safety
    ///
    /// Obtaining a raw pointer to the binary data like this is very unsafe,
    /// and is intended for use cases where low-level access is needed.
    ///
    /// Instead, you should prefer the use of `as_bytes`, which is available
    /// on binary types which implement `AlignedBinary` and which is totally
    /// safe. For bitstrings specifically, you should use one of the iterators
    /// supplied by the `iter` module.
    unsafe fn as_byte_ptr(&self) -> *mut u8;
}

impl<T: ?Sized + Bitstring> Bitstring for Boxed<T> {
    #[inline]
    default fn full_byte_len(&self) -> usize {
        self.as_ref().full_byte_len()
    }

    #[inline]
    default unsafe fn as_byte_ptr(&self) -> *mut u8 {
        self.as_ref().as_byte_ptr()
    }
}


/// This trait provides common behavior for all binary types which represent a collection of bytes
pub trait Binary: Bitstring {
    /// Returns the set of flags that apply to this binary
    fn flags(&self) -> &BinaryFlags;

    /// Returns true if this binary is a raw binary
    #[inline]
    fn is_raw(&self) -> bool {
        self.flags().is_raw()
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    fn is_latin1(&self) -> bool {
        self.flags().is_latin1()
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    fn is_utf8(&self) -> bool {
        self.flags().is_utf8()
    }

    /// Returns a `Encoding` representing the encoding type of this binary
    #[inline]
    fn encoding(&self) -> Encoding {
        self.flags().as_encoding()
    }

    /// Returns the size of just the binary data of this HeapBin in bytes
    #[inline]
    fn size(&self) -> usize {
        self.flags().get_size()
    }
}

impl<T: ?Sized + Binary> Binary for Boxed<T> {
    default fn flags(&self) -> &BinaryFlags {
        self.as_ref().flags()
    }
}

/// Implementors of this trait allow indexing their underlying bytes directly.
///
/// It is expected that an implementation performs such indexing in constant-time
pub trait IndexByte {
    /// Returns the byte found at the given index
    fn byte(&self, index: usize) -> u8;
}

/// This struct represents three pieces of information about a binary:
///
/// - The type of encoding, i.e. latin1, utf8, or unknown/raw
/// - Whether the binary data was compiled in as a literal, and so
///   should never be garbage collected/freed
/// - The size in bytes of binary data, which is used to reify fat
///   pointers to the underlying slice of bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFlags(usize);
impl BinaryFlags {
    // We use the lowest 3 bits to store flags, as this
    // coincidentally allows us to disambiguate `ProcBin`
    // and `BinaryLiteral` without having to know the type
    // in advance. This works because the layouts of both
    // structs start out structurally identical, but while
    // the flags field of `BinaryLiteral` is a usize, the
    // field at that same offset in `ProcBin` is actually
    // a pointer to its `ProcBinInner` companion type. Since
    // we know that all pointers have a minimum alignment
    // of at least 8, the lowest 3 bits are always zero in
    // `ProcBin`, and non-zero in `BinaryLiteral` and all
    // other binary types
    //
    // Note the order here: we're using a more compressed
    // format, so rather than 1 bit per flag, we're counting on
    // the fact that values 1-3 use only the lowest 2 bits,
    // while value 4 is the only value which uses the 3rd bit.
    // This means we can distinguish the binary type flags from
    // the literal flag, allowing a type to have both at the same
    // time, while allowing us to avoid wasting a bit for values
    // which are stored in the remaining bits of the flags field
    const FLAG_BITS: usize = 3;
    const FLAG_IS_RAW_BIN: usize = 1;
    const FLAG_IS_LATIN1_BIN: usize = 2;
    const FLAG_IS_UTF8_BIN: usize = 3;
    const FLAG_IS_LITERAL: usize = 4;
    #[allow(unused)]
    const FLAG_MASK: usize = 0b111;
    const BIN_TYPE_MASK: usize = 0b011;

    /// Converts an `Encoding` to a raw flags bitset
    #[inline]
    pub fn new(encoding: Encoding) -> Self {
        match encoding {
            Encoding::Raw => Self(Self::FLAG_IS_RAW_BIN),
            Encoding::Latin1 => Self(Self::FLAG_IS_LATIN1_BIN),
            Encoding::Utf8 => Self(Self::FLAG_IS_UTF8_BIN),
        }
    }

    /// Converts an `Encoding` to a raw flags bitset for a binary literal
    #[inline]
    pub fn new_literal(encoding: Encoding) -> Self {
        match encoding {
            Encoding::Raw => Self(Self::FLAG_IS_LITERAL + Self::FLAG_IS_RAW_BIN),
            Encoding::Latin1 => Self(Self::FLAG_IS_LITERAL + Self::FLAG_IS_LATIN1_BIN),
            Encoding::Utf8 => Self(Self::FLAG_IS_LITERAL + Self::FLAG_IS_UTF8_BIN),
        }
    }

    #[inline]
    pub fn as_encoding(&self) -> Encoding {
        match self.0 & Self::BIN_TYPE_MASK {
            Self::FLAG_IS_RAW_BIN => Encoding::Raw,
            Self::FLAG_IS_LATIN1_BIN => Encoding::Latin1,
            Self::FLAG_IS_UTF8_BIN => Encoding::Utf8,
            _ => unreachable!()
        }
    }

    #[inline]
    pub fn set_size(self, size: usize) -> Self {
        assert!(size <= (usize::max_value() << Self::FLAG_BITS), "binary size is too large!");
        Self(self.0 << Self::FLAG_BITS)
    }

    #[inline]
    pub fn get_size(&self) -> usize {
        self.0 >> Self::FLAG_BITS
    }

    #[inline]
    pub fn is_literal(&self) -> bool {
        self.0 & Self::FLAG_IS_LITERAL == Self::FLAG_IS_LITERAL
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.0 & Self::BIN_TYPE_MASK == Self::FLAG_IS_RAW_BIN
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.0 & Self::BIN_TYPE_MASK == Self::FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.0 & Self::BIN_TYPE_MASK == Self::FLAG_IS_UTF8_BIN
    }

    #[inline]
    pub fn as_u64(&self) -> u64 {
        let size = (self.0 >> Self::FLAG_BITS) as u64;
        let flags = (self.0 & !Self::BIN_TYPE_MASK) as u64;
        (size << (Self::FLAG_BITS as u64)) | flags
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        let size = (self.0 >> Self::FLAG_BITS) as u32;
        let flags = (self.0 & !Self::BIN_TYPE_MASK) as u32;
        (size << (Self::FLAG_BITS as u32)) | flags
    }
}

/// This trait provides common behavior for bitstrings which are possibly byte-aligned,
/// i.e. the number of bits in the bitstring may or may not be evenly divisible by 8 and
/// does not start at an unaligned bit offset.
pub trait MaybePartialByte {
    /// The number of bits in the partial byte.
    fn partial_byte_bit_len(&self) -> u8;

    /// The total number of bits include those in bytes and any bits in a partial byte.
    fn total_bit_len(&self) -> usize;

    /// The total of number of bytes needed to hold `total_bit_len`
    fn total_byte_len(&self) -> usize;
}


/// Represents an error converting a binary term to `Vec<u8>`
#[derive(Error, Debug)]
pub enum BytesFromBinaryError {
    #[error("unable to allocate memory for binary")]
    Alloc(#[from] Alloc),
    #[error("not a binary value")]
    NotABinary,
    #[error("expected binary term, but got another type")]
    Type,
}

/// Represents an error converting a binary term to `&str`
#[derive(Error, Debug)]
pub enum StrFromBinaryError {
    #[error("unable to allocate memory for binary")]
    Alloc(#[from] Alloc),
    #[error("not a binary value")]
    NotABinary,
    #[error("expected binary term, but got another type")]
    Type,
    #[error("invalid utf-8 encoding")]
    Utf8Error(#[from] Utf8Error),
}

impl From<BytesFromBinaryError> for StrFromBinaryError {
    fn from(bytes_from_binary_error: BytesFromBinaryError) -> StrFromBinaryError {
        use BytesFromBinaryError::*;

        match bytes_from_binary_error {
            Alloc(alloc_err) => StrFromBinaryError::Alloc(alloc_err),
            NotABinary => StrFromBinaryError::NotABinary,
            Type => StrFromBinaryError::NotABinary,
        }
    }
}
