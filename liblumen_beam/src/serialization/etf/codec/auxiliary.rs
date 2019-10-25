use std::ops::Range;

use num::bigint::Sign;

use self::convert::TryInto;
use super::*;

pub fn term_into_atom(t: Term) -> Result<Atom, DecodeError> {
    t.try_into().map_err(|t| DecodeError::UnexpectedType {
        value: t,
        expected: "Atom".to_string(),
    })
}
pub fn term_into_pid(t: Term) -> Result<Pid, DecodeError> {
    t.try_into().map_err(|t| DecodeError::UnexpectedType {
        value: t,
        expected: "Pid".to_string(),
    })
}
pub fn term_into_fix_integer(t: Term) -> Result<FixInteger, DecodeError> {
    t.try_into().map_err(|t| DecodeError::UnexpectedType {
        value: t,
        expected: "FixInteger".to_string(),
    })
}
pub fn term_into_ranged_integer(t: Term, range: Range<i32>) -> Result<i32, DecodeError> {
    term_into_fix_integer(t).and_then(|i| {
        let n = i.value;
        if range.start <= n && n <= range.end {
            Ok(n)
        } else {
            Err(DecodeError::OutOfRange { value: n, range })
        }
    })
}
pub fn invalid_data_error<T>(message: String) -> std::io::Result<T> {
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message,
    ))
}
pub fn other_error<T>(message: String) -> std::io::Result<T> {
    Err(std::io::Error::new(std::io::ErrorKind::Other, message))
}
pub fn latin1_bytes_to_string(buf: &[u8]) -> std::io::Result<String> {
    // FIXME: Supports Latin1 characters
    std::str::from_utf8(buf)
        .or_else(|e| other_error(e.to_string()))
        .map(|s| s.to_string())
}
pub fn byte_to_sign(b: u8) -> std::io::Result<Sign> {
    match b {
        0 => Ok(Sign::Plus),
        1 => Ok(Sign::Minus),
        _ => invalid_data_error(format!("A sign value must be 0 or 1: value={}", b)),
    }
}
pub fn sign_to_byte(sign: Sign) -> u8 {
    if sign == Sign::Minus {
        1
    } else {
        0
    }
}
