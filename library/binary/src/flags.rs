use core::fmt;
use core::str::FromStr;

use anyhow::anyhow;

/// Represents the original encoding of a binary
///
/// In the case of `Raw`, there is no specific encoding and
/// while it may be valid Latin-1 or UTF-8 bytes, it should be
/// treated as neither without validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Raw,
    Latin1,
    Utf8,
    Utf16,
    Utf32,
}
impl Encoding {
    /// Determines the best encoding that fits the given byte slice.
    ///
    /// If the bytes are valid UTF-8, it will be used. Otherwise, the
    /// bytes must either be valid Latin-1 (i.e. ISO/IEC 8859-1) or raw.
    pub fn detect(bytes: &[u8]) -> Self {
        match core::str::from_utf8(bytes) {
            Ok(_) => Self::Utf8,
            Err(_) => {
                if Self::is_latin1(bytes) {
                    Self::Latin1
                } else {
                    Self::Raw
                }
            }
        }
    }

    #[inline]
    pub fn is_latin1(s: &[u8]) -> bool {
        s.iter().copied().all(Self::is_latin1_byte)
    }

    #[inline(always)]
    pub fn is_latin1_byte(byte: u8) -> bool {
        // The Latin-1 codepage starts at 0x20, skips 0x7F-0x9F, then continues to 0xFF
        (byte <= 0x1F) | (0x7F..=0x9F).contains(&byte)
    }
}
impl FromStr for Encoding {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "raw" => Ok(Self::Raw),
            "latin1" => Ok(Self::Latin1),
            "utf8" | "unicode" => Ok(Self::Utf8),
            "utf16" => Ok(Self::Utf16),
            "utf32" => Ok(Self::Utf32),
            other => Err(anyhow!(
                "unrecognized encoding '{}', expected raw, latin1, utf8/unicode, utf16, or utf32",
                other
            )),
        }
    }
}
impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Raw => f.write_str("raw"),
            Self::Latin1 => f.write_str("latin1"),
            Self::Utf8 => f.write_str("utf8"),
            Self::Utf16 => f.write_str("utf16"),
            Self::Utf32 => f.write_str("utf32"),
        }
    }
}
/// This struct represents the following information about a binary/bitstring:
///
/// - The type of encoding, i.e. latin1, utf8, or unknown/raw
/// - Whether it is heap-allocated (small) or reference-counted
/// - Whether it is a binary or a bitstring
/// - If a bitstring, the number of trailing bits in the underlying buffer
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFlags(usize);
impl BinaryFlags {
    const FLAG_IS_RAW_BIN: usize = 0b001;
    const FLAG_IS_LATIN1_BIN: usize = 0b010;
    const FLAG_IS_UTF8_BIN: usize = 0b100;
    const FLAG_IS_BITSTRING: usize = 0b011;
    const FLAG_ENCODING_MASK: usize = 0b111;
    const FLAG_TRAILING_BITS_MASK: usize = 0b1110000;
    const FLAG_TRAILING_BITS_SHIFT: usize = 4;
    const FLAG_SIZE_SHIFT: usize = 7;
    const FLAG_SIZE_MASK: usize = !0b1111111;

    /// Creates a new set of flags based on the given binary data size and encoding
    ///
    /// Use `with_trailing_bits` to mark the data as bitstring as needed.
    #[inline]
    pub const fn new(size: usize, encoding: Encoding) -> Self {
        let size = size << Self::FLAG_SIZE_SHIFT;
        let flags = match encoding {
            Encoding::Raw => size | Self::FLAG_IS_RAW_BIN,
            Encoding::Latin1 => size | Self::FLAG_IS_LATIN1_BIN,
            Encoding::Utf8 => size | Self::FLAG_IS_UTF8_BIN,
            _ => panic!("invalid binary encoding, must be latin1 or utf8"),
        };
        Self(flags)
    }

    #[inline(always)]
    pub const unsafe fn from_raw(raw: usize) -> Self {
        Self(raw)
    }

    #[inline(always)]
    pub const fn into_raw(self) -> usize {
        self.0
    }

    /// Marks the underlying binary data as bitstring data with `n` trailing bits.
    ///
    /// This function will panic if `n` is greater than 7.
    pub const fn with_trailing_bits(self, n: usize) -> Self {
        assert!(n < 8);
        let tb = n << Self::FLAG_TRAILING_BITS_SHIFT;
        Self(self.0 & Self::FLAG_SIZE_MASK | Self::FLAG_IS_BITSTRING | tb)
    }

    /// Returns the number of trailing bits for the bitstring associated with these flags
    #[inline]
    pub const fn trailing_bits(&self) -> usize {
        (self.0 & Self::FLAG_TRAILING_BITS_MASK) >> Self::FLAG_TRAILING_BITS_SHIFT
    }

    /// Returns the encoding of the bytes containined in the underlying data
    #[inline]
    pub fn as_encoding(&self) -> Encoding {
        match self.0 & Self::FLAG_ENCODING_MASK {
            Self::FLAG_IS_RAW_BIN | Self::FLAG_IS_BITSTRING => Encoding::Raw,
            Self::FLAG_IS_LATIN1_BIN => Encoding::Latin1,
            Self::FLAG_IS_UTF8_BIN => Encoding::Utf8,
            value => unreachable!("{}", value),
        }
    }

    /// Returns true if the binary associated with these flags is actually a bitstring
    #[inline]
    pub const fn is_bitstring(&self) -> bool {
        self.0 & Self::FLAG_ENCODING_MASK == Self::FLAG_IS_BITSTRING
    }

    /// Returns the size of the binary in bytes
    #[inline]
    pub const fn size(&self) -> usize {
        let base_size = self.0 >> Self::FLAG_SIZE_SHIFT;
        base_size + (self.trailing_bits() > 0) as usize
    }

    /// Returns true if the binary associated with these flags is under the MAX_HEAP_SIZE limit
    ///
    /// When true, the underlying binary will be heap allocated, otherwise it will be
    /// reference-counted
    #[inline]
    pub fn is_small(&self) -> bool {
        self.size() <= 64
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        matches!(
            self.0 & Self::FLAG_ENCODING_MASK,
            Self::FLAG_IS_RAW_BIN | Self::FLAG_IS_BITSTRING
        )
    }

    /// Returns true if this binary is a Latin-1 binary
    #[inline]
    pub fn is_latin1(&self) -> bool {
        self.0 & Self::FLAG_ENCODING_MASK == Self::FLAG_IS_LATIN1_BIN
    }

    /// Returns true if this binary is a UTF-8 binary
    #[inline]
    pub fn is_utf8(&self) -> bool {
        self.0 & Self::FLAG_ENCODING_MASK == Self::FLAG_IS_UTF8_BIN
    }
}
impl fmt::Debug for BinaryFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BinaryFlags")
            .field("encoding", &format_args!("{}", self.as_encoding()))
            .field("size", &self.size())
            .field("is_small", &self.is_small())
            .field("is_bitstring", &self.is_bitstring())
            .field("trailing_bits", &self.trailing_bits())
            .finish()
    }
}
