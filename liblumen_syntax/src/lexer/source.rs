use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::char;

use failure::Fail;

use crate::util;
use liblumen_diagnostics::{ByteIndex, ByteOffset, ByteSpan, CodeMap, FileMap, FileName, Diagnostic};

pub type SourceResult<T> = std::result::Result<T, SourceError>;

pub trait Source: Sized {
    fn from_path<P: AsRef<Path>>(codemap: Arc<Mutex<CodeMap>>, path: P)
        -> SourceResult<Self>;

    fn read(&mut self) -> Option<(ByteIndex, char)>;

    fn peek(&mut self) -> Option<(ByteIndex, char)>;

    fn span(&self) -> ByteSpan;

    fn slice(&self, span: ByteSpan) -> &str;
}

#[derive(Fail, Debug)]
pub enum SourceError {
    #[fail(display = "{}", _0)]
    IO(std::io::Error),

    #[fail(display = "invalid source path")]
    InvalidPath(String),

    #[fail(display = "invalid environment variable '{}'", _1)]
    InvalidEnvironmentVariable(std::env::VarError, String)
}
impl std::convert::From<std::io::Error> for SourceError {
    fn from(err: std::io::Error) -> SourceError {
        SourceError::IO(err)
    }
}
impl SourceError {
    pub fn to_diagnostic(&self) -> Diagnostic {
        match self {
            SourceError::IO(ref err) =>
                Diagnostic::new_error(err.to_string()),
            SourceError::InvalidPath(ref reason) =>
                Diagnostic::new_error(format!("invalid path: {}", reason)),
            SourceError::InvalidEnvironmentVariable(env::VarError::NotPresent, ref var) =>
                Diagnostic::new_error(format!("invalid environment variable '{}': not defined", var)),
            SourceError::InvalidEnvironmentVariable(_, ref var) =>
                Diagnostic::new_error(format!("invalid environment variable '{}': contains invalid unicode data", var)),
        }
    }
}

/// A source which reads from a `diagnostics::FileMap`
pub struct FileMapSource {
    src: Arc<FileMap>,
    bytes: *const [u8],
    start: ByteIndex,
    peek: Option<(ByteIndex, char)>,
    end: usize,
    pos: usize,
    eof: bool,
}
impl FileMapSource {
    pub fn new(src: Arc<FileMap>) -> Self {
        let start = src.span().start();
        let end = src.src.len();
        let bytes: *const [u8] = src.src.as_bytes();
        let mut source = FileMapSource {
            src,
            bytes,
            peek: None,
            start,
            end,
            pos: 0,
            eof: false,
        };
        source.peek = unsafe { source.next_char_internal() };
        source
    }

    fn peek_char(&self) -> Option<(ByteIndex, char)> {
        self.peek
    }

    fn next_char(&mut self) -> Option<(ByteIndex, char)> {
        // If we've peeked a char already, return that
        let result = if self.peek.is_some() {
            std::mem::replace(&mut self.peek, None)
        } else {
            let next = unsafe { self.next_char_internal() };
            match next {
                None => {
                    self.eof = true;
                    return None;
                }
                result => result
            }
        };

        // Reset peek
        self.peek = unsafe { self.next_char_internal() };

        result
    }

    #[inline]
    unsafe fn next_char_internal(&mut self) -> Option<(ByteIndex, char)> {
        let mut pos = self.pos;
        let end = self.end;
        if pos == end {
            self.eof = true;
        }

        if self.eof {
            return None;
        }

        let start = self.start + ByteOffset(pos as i64);

        let bytes: &[u8] = &*self.bytes;

        // Decode UTF-8
        let x = *bytes.get_unchecked(pos);
        if x < 128 {
            self.pos = pos + 1;
            return Some((start, char::from_u32_unchecked(x as u32)))
        }

        // Multibyte case follows
        // Decode from a byte combination out of: [[[x y] z] w]
        // NOTE: Performance is sensitive to the exact formulation here
        let init = Self::utf8_first_byte(x, 2);

        pos = pos + 1;
        let y = if pos == end {
            0u8
        } else {
            *bytes.get_unchecked(pos)
        };
        let mut ch = Self::utf8_acc_cont_byte(init, y);
        if x >= 0xE0 {
            // [[x y z] w] case
            // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
            pos = pos + 1;
            let z = if pos == end {
                0u8
            } else {
                *bytes.get_unchecked(pos)
            };
            let y_z = Self::utf8_acc_cont_byte((y & Self::CONT_MASK) as u32, z);
            ch = init << 12 | y_z;
            if x >= 0xF0 {
                // [x y z w] case
                // use only the lower 3 bits of `init`
                pos = pos + 1;
                let w = if pos == end {
                    0u8
                } else {
                    *bytes.get_unchecked(pos)
                };
                ch = (init & 7) << 18 | Self::utf8_acc_cont_byte(y_z, w);
            }
        }

        pos = pos + 1;
        if pos >= end {
            self.eof = true
        }
        self.pos = pos;

        Some((start, char::from_u32_unchecked(ch as u32)))
    }

    /// Returns the initial codepoint accumulator for the first byte.
    /// The first byte is special, only want bottom 5 bits for width 2, 4 bits
    /// for width 3, and 3 bits for width 4.
    #[inline]
    fn utf8_first_byte(byte: u8, width: u32) -> u32 { (byte & (0x7F >> width)) as u32 }

    /// Returns the value of `ch` updated with continuation byte `byte`.
    #[inline]
    fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 { (ch << 6) | (byte & Self::CONT_MASK) as u32 }

    /// Mask of the value bits of a continuation byte.
    const CONT_MASK: u8 = 0b0011_1111;
}
impl Source for FileMapSource {
    fn from_path<P: AsRef<Path>>(codemap: Arc<Mutex<CodeMap>>, path: P)
        -> SourceResult<Self>
    {
        let path = util::substitute_path_variables(path)?;
        let filemap = {
            codemap.lock()
                .unwrap()
                .add_filemap_from_disk(FileName::real(path))?
        };
        Ok(FileMapSource::new(filemap))
    }

    #[inline]
    fn read(&mut self) -> Option<(ByteIndex, char)> {
        self.next_char()
    }

    #[inline]
    fn peek(&mut self) -> Option<(ByteIndex, char)> {
        self.peek_char()
    }

    #[inline]
    fn span(&self) -> ByteSpan {
        ByteSpan::new(self.start.clone(), self.start + ByteOffset(self.end as i64))
    }

    #[inline]
    fn slice(&self, span: ByteSpan) -> &str {
        self.src.src_slice(span).unwrap()
    }
}

impl Iterator for FileMapSource {
    type Item = (ByteIndex, char);

    fn next(&mut self) -> Option<Self::Item> {
        self.read()
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;
    use liblumen_diagnostics::{FileMap, FileName, ByteIndex};

    use super::*;

    fn make_source(input: &str) -> FileMapSource {
        let filemap = FileMap::new(FileName::Virtual(Cow::Borrowed("nofile")), input);
        FileMapSource::new(Arc::new(filemap))
    }

    fn read_all_chars(input: &str) -> Vec<char> {
        let source = make_source(input);
        source.map(|result| result.1).collect()
    }

    #[test]
    fn file_source() {
        let expected = vec![
            'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'
        ];

        let chars = read_all_chars("hello world!");

        assert_eq!(expected, chars);

        let mut source = make_source("hello world!");
        assert_eq!(Some((ByteIndex(1), 'h')), source.peek());
        assert_eq!(Some((ByteIndex(1), 'h')), source.next());

        let mut source = make_source("éé");
        assert_eq!(Some((ByteIndex(1), 'é')), source.peek());
        assert_eq!(Some((ByteIndex(1), 'é')), source.next());
        assert_eq!(Some((ByteIndex(3), 'é')), source.peek());
        assert_eq!(Some((ByteIndex(3), 'é')), source.next());
    }
}
