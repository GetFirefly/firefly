use static_assertions::assert_eq_size;

use crate::Endianness;

/// Represents a binary segment constructor/match specification, e.g. `<<42:8/signed-little-integer>>`
///
/// The size and layout of this type is relied upon by our compiler. It can be represented as a single i64
/// value, and has a layout equivalent to the following struct:
///
/// ```rust,ignore
/// #[repr(C)]
/// pub struct BinaryEntrySpecifier {
///   tag: u32,
///   data: [u32; 1]
/// }
/// ```
///
/// Where individual variants of the enum are discriminated by the tag and are encoded like:
///
/// ```rust,ignore
/// #[repr(C)]
/// pub struct BinaryEntrySpecifierInteger {
///   _tag: [i8; 4],
///   signed: u8,
///   endianness: u8,
///   unit: u8,
///   _padding: [i8; 1]
/// }
/// ```
///
/// NOTE: This compact encoding is possible because `Endianness` is `#[repr(u8)]`
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum BinaryEntrySpecifier {
    Integer {
        signed: bool,
        endianness: Endianness,
        unit: u8,
    } = 0,
    Float {
        endianness: Endianness,
        unit: u8,
    } = 1,
    Binary {
        unit: u8,
    } = 2,
    Utf8 = 3,
    Utf16 {
        endianness: Endianness,
    } = 4,
    Utf32 {
        endianness: Endianness,
    } = 5,
}

assert_eq_size!(BinaryEntrySpecifier, u32);

impl BinaryEntrySpecifier {
    pub const DEFAULT: Self = Self::Integer {
        signed: false,
        endianness: Endianness::Big,
        unit: 1,
    };

    pub fn is_float(&self) -> bool {
        match self {
            Self::Float { .. } => true,
            _ => false,
        }
    }

    pub fn unit(&self) -> usize {
        match self {
            Self::Integer { unit, .. } | Self::Float { unit, .. } | Self::Binary { unit, .. } => {
                *unit as usize
            }
            _ => 1,
        }
    }

    pub fn has_size(&self) -> bool {
        match self {
            Self::Utf8 => false,
            Self::Utf16 { .. } => false,
            Self::Utf32 { .. } => false,
            _ => true,
        }
    }

    pub fn is_native_endian(&self) -> bool {
        match self {
            Self::Integer {
                endianness: Endianness::Native,
                ..
            } => true,
            Self::Float {
                endianness: Endianness::Native,
                ..
            } => true,
            Self::Utf16 {
                endianness: Endianness::Native,
                ..
            } => true,
            Self::Utf32 {
                endianness: Endianness::Native,
                ..
            } => true,
            _ => false,
        }
    }
}
impl Default for BinaryEntrySpecifier {
    fn default() -> Self {
        Self::DEFAULT
    }
}
