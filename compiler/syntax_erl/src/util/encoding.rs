#![allow(dead_code, unused_variables)]
use std::fmt::{Display, Formatter};

use liblumen_binary::{BitCarrier, BitRead, Endianness};
use liblumen_diagnostics::{Diagnostic, Label, SourceSpan, ToDiagnostic};

#[derive(Debug, thiserror::Error)]
pub enum StringError {
    #[error("unicode codepoint #{codepoint} is not encodable in {encoding:?}")]
    CodepointEncoding {
        span: SourceSpan,
        codepoint: u64,
        encoding: Encoding,
    },
}

impl ToDiagnostic for StringError {
    fn to_diagnostic(&self) -> Diagnostic {
        let msg = self.to_string();
        match self {
            StringError::CodepointEncoding {
                span,
                codepoint,
                encoding,
            } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), *span)
                    .with_message("encoding failed at codepoint")]),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Encoding {
    /// The default encoding for binary string literals.
    /// In practice this is the unicode codepoint modulo 2^8 (truncated to 8 bits).
    Latin1,
    Utf8,
    Utf16,
    Utf32,
}

impl Display for Encoding {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            Encoding::Latin1 => write!(f, "latin1"),
            Encoding::Utf8 => write!(f, "utf8"),
            Encoding::Utf16 => write!(f, "utf16"),
            Encoding::Utf32 => write!(f, "utf32"),
        }
    }
}

impl Encoding {
    pub fn encode(&self, cp: u64, span: SourceSpan) -> Result<Encoded, StringError> {
        match self {
            Encoding::Latin1 => encode_latin1(cp, span),
            Encoding::Utf8 => encode_utf8(cp, span),
            Encoding::Utf16 => encode_utf16(cp, span),
            Encoding::Utf32 => encode_utf32(cp, span),
        }
    }
}

#[derive(Copy, Clone)]
pub enum Encoded {
    N1(u8),
    N2(u8, u8),
    N3(u8, u8, u8),
    N4(u8, u8, u8, u8),
}
impl Encoded {
    pub fn swap(self) -> Self {
        match self {
            Encoded::N1(a) => Encoded::N1(a),
            Encoded::N2(a, b) => Encoded::N2(b, a),
            Encoded::N3(a, b, c) => Encoded::N3(c, b, a),
            Encoded::N4(a, b, c, d) => Encoded::N4(d, c, b, a),
        }
    }

    pub fn write(self, endianness: Endianness, out: &mut Vec<u8>) {
        match endianness {
            Endianness::Big => self.write_be(out),
            Endianness::Little => self.write_le(out),
            Endianness::Native => {
                if cfg!(target_endian = "big") {
                    self.write_be(out)
                } else {
                    self.write_le(out)
                }
            }
        }
    }

    pub fn write_le(self, out: &mut Vec<u8>) {
        match self {
            Encoded::N1(a) => out.push(a),
            Encoded::N2(a, b) => {
                out.push(b);
                out.push(a);
            }
            Encoded::N3(a, b, c) => {
                out.push(c);
                out.push(b);
                out.push(a);
            }
            Encoded::N4(a, b, c, d) => {
                out.push(d);
                out.push(c);
                out.push(b);
                out.push(a);
            }
        }
    }

    pub fn write_be(self, out: &mut Vec<u8>) {
        match self {
            Encoded::N1(a) => out.push(a),
            Encoded::N2(a, b) => {
                out.push(a);
                out.push(b);
            }
            Encoded::N3(a, b, c) => {
                out.push(a);
                out.push(b);
                out.push(c);
            }
            Encoded::N4(a, b, c, d) => {
                out.push(a);
                out.push(b);
                out.push(c);
                out.push(d);
            }
        }
    }
}
impl BitCarrier for Encoded {
    type T = u8;
    fn bit_len(&self) -> usize {
        match self {
            Self::N1(_) => 8,
            Self::N2(_, _) => 16,
            Self::N3(_, _, _) => 24,
            Self::N4(_, _, _, _) => 32,
        }
    }
}
impl BitRead for Encoded {
    fn read_word(&self, n: usize) -> u8 {
        match (self, n) {
            (Self::N1(val), 0) => *val,
            (Self::N2(val, _), 0) => *val,
            (Self::N2(_, val), 1) => *val,
            (Self::N3(val, _, _), 0) => *val,
            (Self::N3(_, val, _), 1) => *val,
            (Self::N3(_, _, val), 2) => *val,
            (Self::N4(val, _, _, _), 0) => *val,
            (Self::N4(_, val, _, _), 1) => *val,
            (Self::N4(_, _, val, _), 2) => *val,
            (Self::N4(_, _, _, val), 3) => *val,
            _ => unreachable!(),
        }
    }
}

pub fn encode_utf8(cp: u64, span: SourceSpan) -> Result<Encoded, StringError> {
    match cp {
        0x00..=0x7f => Ok(Encoded::N1(cp as u8)),
        0x80..=0x7ff => Ok(Encoded::N2(
            0b110_00000 | (cp >> 6 & 0b000_11111) as u8,
            0b10_000000 | (cp >> 0 & 0b00_111111) as u8,
        )),
        0x800..=0xffff => Ok(Encoded::N3(
            0b1110_0000 | (cp >> 12 & 0b0000_1111) as u8,
            0b10_000000 | (cp >> 6 & 0b00_111111) as u8,
            0b10_000000 | (cp >> 0 & 0b00_111111) as u8,
        )),
        0x10000..=0x1fffff => Ok(Encoded::N4(
            0b11110_000 | (cp >> 18 & 0b00000_111) as u8,
            0b10_000000 | (cp >> 12 & 0b00_111111) as u8,
            0b10_000000 | (cp >> 6 & 0b00_111111) as u8,
            0b10_000000 | (cp >> 0 & 0b00_111111) as u8,
        )),
        _ => Err(StringError::CodepointEncoding {
            span,
            codepoint: cp,
            encoding: Encoding::Utf8,
        }),
    }
}

pub fn encode_utf16(cp: u64, span: SourceSpan) -> Result<Encoded, StringError> {
    match cp {
        0x0000..=0xd7ff => Ok(Encoded::N2((cp >> 8) as u8, (cp >> 0) as u8)),
        0xd800..=0xdfff => Err(StringError::CodepointEncoding {
            span,
            codepoint: cp,
            encoding: Encoding::Utf16,
        }),
        0xd800..=0xffff => Ok(Encoded::N2((cp >> 8) as u8, (cp >> 0) as u8)),
        0x10000..=0x10ffff => {
            let val = cp - 0x10000;
            Ok(Encoded::N4(
                0b110110_00 | (cp >> 18 & 0b000000_11) as u8,
                (cp >> 10) as u8,
                0b110111_00 | (cp >> 8 & 0b000000_11) as u8,
                (cp >> 0) as u8,
            ))
        }
        _ => Err(StringError::CodepointEncoding {
            span,
            codepoint: cp,
            encoding: Encoding::Utf16,
        }),
    }
}

pub fn encode_utf32(cp: u64, span: SourceSpan) -> Result<Encoded, StringError> {
    if cp > std::u32::MAX as u64 {
        Err(StringError::CodepointEncoding {
            span,
            codepoint: cp,
            encoding: Encoding::Utf32,
        })
    } else {
        Ok(Encoded::N4(
            (cp >> 24) as u8,
            (cp >> 16) as u8,
            (cp >> 8) as u8,
            (cp >> 0) as u8,
        ))
    }
}

fn encode_latin1(cp: u64, _span: SourceSpan) -> Result<Encoded, StringError> {
    Ok(Encoded::N1(cp as u8))
}
