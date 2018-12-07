use std::borrow::Cow;
use std::char;
use std::iter::Peekable;

use num::Num;

use trackable::{track, track_assert_some, track_panic};

use super::{Error, ErrorKind, Result};

pub fn is_atom_head_char(c: char) -> bool {
    if let 'a'...'z' = c {
        true
    } else {
        c.is_lowercase() && c.is_alphabetic()
    }
}

pub fn is_atom_non_head_char(c: char) -> bool {
    match c {
        '@' | '_' | '0'...'9' => true,
        _ => c.is_alphabetic(),
    }
}

pub fn is_variable_head_char(c: char) -> bool {
    match c {
        'A'...'Z' | '_' => true,
        _ => false,
    }
}

pub fn is_variable_non_head_char(c: char) -> bool {
    match c {
        'a'...'z' | 'A'...'Z' | '@' | '_' | '0'...'9' => true,
        _ => false,
    }
}

pub fn parse_string(input: &str, terminator: char) -> Result<(Cow<str>, usize)> {
    let maybe_end = track_assert_some!(input.find(terminator), ErrorKind::InvalidInput);
    let maybe_escaped = unsafe { input.get_unchecked(0..maybe_end).contains('\\') };
    if maybe_escaped {
        let (s, end) = track!(parse_string_owned(input, terminator))?;
        Ok((Cow::Owned(s), end))
    } else {
        let slice = unsafe { input.get_unchecked(0..maybe_end) };
        Ok((Cow::Borrowed(slice), maybe_end))
    }
}

fn parse_string_owned(input: &str, terminator: char) -> Result<(String, usize)> {
    let mut buf = String::new();
    let mut chars = input.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '\\' {
            let c = track!(parse_escaped_char(&mut chars))?;
            buf.push(c);
        } else if c == terminator {
            return Ok((buf, i));
        } else {
            buf.push(c);
        }
    }
    track_panic!(ErrorKind::UnexpectedEos);
}

// http://erlang.org/doc/reference_manual/data_types.html#id76758
pub fn parse_escaped_char<I>(chars: &mut Peekable<I>) -> Result<char>
where
    I: Iterator<Item = (usize, char)>,
{
    let (_, c) = track_assert_some!(chars.next(), ErrorKind::UnexpectedEos);
    match c {
        'b' => Ok(8 as char),   // Back Space
        'd' => Ok(127 as char), // Delete
        'e' => Ok(27 as char),  // Escape,
        'f' => Ok(12 as char),  // Form Feed
        'n' => Ok('\n'),
        'r' => Ok('\r'),
        's' => Ok(' '),
        't' => Ok('\t'),
        'v' => Ok(11 as char), // Vertical Tabulation
        '^' => {
            let (_, c) = track_assert_some!(chars.next(), ErrorKind::UnexpectedEos);
            Ok((c as u32 % 32) as u8 as char)
        }
        'x' => {
            let (_, c) = track_assert_some!(chars.next(), ErrorKind::UnexpectedEos);
            let buf = if c == '{' {
                chars.map(|(_, c)| c).take_while(|c| *c != '}').collect()
            } else {
                let mut buf = String::with_capacity(2);
                buf.push(c);
                buf.push(track_assert_some!(
                    chars.next().map(|(_, c)| c),
                    ErrorKind::UnexpectedEos
                ));
                buf
            };
            let code: u32 = track!(Num::from_str_radix(&buf, 16).map_err(Error::from))?;
            Ok(track_assert_some!(
                char::from_u32(code),
                ErrorKind::InvalidInput
            ))
        }
        c @ '0'...'7' => {
            let mut limit = 2;
            let mut n = c.to_digit(8).expect("Never fails");
            while let Some((_, '0'...'7')) = chars.peek().cloned() {
                n = (n * 8) + c.to_digit(8).expect("Never fails");
                let _ = chars.next();
                limit -= 1;
                if limit == 0 {
                    break;
                }
            }
            Ok(track_assert_some!(
                char::from_u32(n),
                ErrorKind::InvalidInput
            ))
        }
        _ => Ok(c),
    }
}
