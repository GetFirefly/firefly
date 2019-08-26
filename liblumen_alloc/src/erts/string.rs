use core::fmt;
use core::str;
use alloc::string::String;

/// Represents the original encoding of a binary
///
/// In the case of `Raw`, there is no specific encoding and
/// while it may be valid Latin-1 or UTF-8 bytes, it should be
/// treated as neither without validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Raw,
    Latin1,
    Utf8
}
impl Encoding {
    pub fn from_str(s: &str) -> Self {
        if s.is_ascii() {
            Self::Latin1
        } else {
            Self::Utf8
        }
    }
}

/// Represents the direction encoding is performed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    ToBytes,
    ToString
}

/// Represents an error which occurs when converting a string to bytes
/// in a given encoding; or vice versa, decoding bytes to a string.
#[derive(Debug)]
pub struct InvalidEncodingError {
    code: u16,
    index: usize,
    encoding: Encoding,
    direction: Direction,
}
impl InvalidEncodingError {
    fn new(code: u16, index: usize, encoding: Encoding, direction: Direction) -> Self {
        Self {
            code,
            index,
            encoding,
            direction,
        }
    }
}
impl fmt::Display for InvalidEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let to_bytes = self.direction == Direction::ToBytes;
        match self.encoding {
            Encoding::Latin1 if to_bytes => {
                write!(f, "cannot encode string as latin-1 bytes (character code = {}, index = {})", self.code, self.index)
            }
            Encoding::Utf8 => {
                write!(f, "cannot decode bytes to UTF-8 string (character code = {}, index = {})", self.code, self.index)
            }
            // It is never possible to fail decoding a slice of u8 to latin-1,
            // or encoding Rust strings as UTF-8 bytes
            _ => unreachable!()
        }
    }
}

/// Returns true if the given `str` is encodable as latin-1 bytes
pub fn is_latin1(s: &str) -> bool {
    s.chars().all(|c| c as u16 <= 255)
}

/// Converts a Latin-1 encoded binary slice to a `String`
pub fn to_latin1_string(bytes: &[u8]) -> String {
    bytes.iter().copied().map(|b| { b as char }).collect()
}

/// Converts a `str` to valid Latin-1 bytes, if composed of Latin-1 encodable characters
///
/// Returns `InvalidEncodingError` if this `str` is not encodable as Latin-1
pub fn to_latin1_bytes(s: &str) -> Result<Vec<u8>, InvalidEncodingError> {
    let mut bytes = Vec::with_capacity(s.len());
    for (index, c) in s.char_indices() {
        let code = c as u16;
        if code > 255 {
            return Err(InvalidEncodingError::new(code, index, Encoding::Latin1, Direction::ToBytes))
        }
        bytes.push(code as u8);
    }
    Ok(bytes)
}

/// Converts a UTF-8 encoded binary slice to a `str`
///
/// Returns `Ok(str)` if successful, otherwise `Err(InvalidEncodingError)`
pub fn as_utf8_str(bytes: &[u8]) -> Result<&str, InvalidEncodingError> {
    str::from_utf8(bytes).map_err(|err| {
        let index = err.valid_up_to();
        let code = bytes[index] as u16;
        InvalidEncodingError::new(code, index, Encoding::Utf8, Direction::ToString)
    })
}

/// Converts a UTF-8 encoded binary slice to a `String`
///
/// Returns `Ok(String)` if successful, otherwise `Err(InvalidEncodingError)`
pub fn to_utf8_string(bytes: &[u8]) -> Result<String, InvalidEncodingError> {
    String::from_utf8(bytes.to_vec()).map_err(|err| {
        let index = err.utf8_error().valid_up_to();
        let code = bytes[index] as u16;
        InvalidEncodingError::new(code, index, Encoding::Utf8, Direction::ToString)
    })
}
