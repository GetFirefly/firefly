use core::fmt;
use core::str::FromStr;

use anyhow::anyhow;

use crate::term::{Atom, Term};

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
        s.iter().copied().all(|b| is_latin1_byte(b))
    }

    #[inline(always)]
    pub fn is_latin1_byte(byte: u8) -> bool {
        // The Latin-1 codepage starts at 0x20, skips 0x7F-0x9F, then continues to 0xFF
        (digit <= 0x1F) | ((digit >= 0x7F) & (digit <= 0x9F))
    }
}
impl FromStr for Encoding {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Error> {
        match s {
            "raw" => Ok(Self::Raw),
            "latin1" => Ok(Self::Latin1),
            "utf8" => Ok(Self::Utf8),
            other => Err(anyhow!(
                "unrecognized encoding '{}', expected raw, latin1, or utf8",
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
        }
    }
}
// Support converting from atom terms to `Encoding` type
impl TryFrom<Term> for Encoding {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Atom(a) => a.as_str().parse(),
            other => Err(anyhow!(
                "invalid encoding name: expected atom; got {}",
                other.type_of()
            )),
        }
    }
}
// Support converting from atom terms to `Encoding` type
impl TryFrom<Atom> for Encoding {
    type Error = InvalidEncodingNameError;

    #[inline]
    fn try_from(atom: Atom) -> Result<Self, Self::Error> {
        atom.as_str().parse()
    }
}

/// This struct represents two pieces of information about a binary:
///
/// - The type of encoding, i.e. latin1, utf8, or unknown/raw
/// - Whether the binary data was compiled in as a literal, and so should never be garbage
///   collected/freed
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BinaryFlags(usize);
impl BinaryFlags {
    const FLAG_IS_RAW_BIN: usize = 1;
    const FLAG_IS_LATIN1_BIN: usize = 1 << 1;
    const FLAG_IS_UTF8_BIN: usize = 1 << 2;
    const FLAG_IS_LITERAL: usize = 1 << 3;
    const FLAG_ENCODING_MASK: usize = 0b111;

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
            Encoding::Raw => Self(Self::FLAG_IS_LITERAL | Self::FLAG_IS_RAW_BIN),
            Encoding::Latin1 => Self(Self::FLAG_IS_LITERAL | Self::FLAG_IS_LATIN1_BIN),
            Encoding::Utf8 => Self(Self::FLAG_IS_LITERAL | Self::FLAG_IS_UTF8_BIN),
        }
    }

    #[inline]
    pub fn as_encoding(&self) -> Encoding {
        match self.0 & Self::FLAG_ENCODING_MASK {
            Self::FLAG_IS_RAW_BIN => Encoding::Raw,
            Self::FLAG_IS_LATIN1_BIN => Encoding::Latin1,
            Self::FLAG_IS_UTF8_BIN => Encoding::Utf8,
            value => unreachable!("{}", value),
        }
    }

    #[inline]
    pub fn is_literal(&self) -> bool {
        self.0 & Self::FLAG_IS_LITERAL == Self::FLAG_IS_LITERAL
    }

    /// Returns true if this binary is a raw binary
    #[inline]
    pub fn is_raw(&self) -> bool {
        self.0 & Self::FLAG_ENCODING_MASK == Self::FLAG_IS_RAW_BIN
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
            .field("literal", &self.is_literal())
            .finish()
    }
}
