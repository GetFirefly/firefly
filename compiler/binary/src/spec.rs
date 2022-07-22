use crate::Endianness;

/// Represents a binary segment constructor/match specification, e.g. `<<42:8/signed-little-integer>>`
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BinaryEntrySpecifier {
    Integer {
        signed: bool,
        endianness: Endianness,
        unit: usize,
    },
    Float {
        endianness: Endianness,
        unit: usize,
    },
    Binary {
        unit: usize,
    },
    Utf8,
    Utf16 {
        endianness: Endianness,
    },
    Utf32 {
        endianness: Endianness,
    },
}
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
                *unit
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
