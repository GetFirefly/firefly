use std::char;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use firefly_util::diagnostics::*;

pub type SourceResult<T> = std::result::Result<T, SourceError>;

pub trait Source: Sized {
    fn new(src: Arc<SourceFile>) -> Self;

    fn read(&mut self) -> Option<(SourceIndex, char)>;

    fn peek(&mut self) -> Option<(SourceIndex, char)>;

    fn span(&self) -> SourceSpan;

    fn slice(&self, span: impl Into<Range<usize>>) -> &str;
}

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("error reading {path:?}: {source:?}")]
    RootFileIO {
        source: std::io::Error,
        path: PathBuf,
    },

    #[error("invalid source path")]
    InvalidPath { reason: String },

    #[error(transparent)]
    PathVariableSubstitute {
        #[from]
        source: crate::util::PathVariableSubstituteError,
    },
}
impl ToDiagnostic for SourceError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            SourceError::RootFileIO { source, path: _ } => {
                Diagnostic::error().with_message(source.to_string())
            }
            SourceError::InvalidPath { reason } => {
                Diagnostic::error().with_message(format!("invalid path: {}", reason))
            }
            SourceError::PathVariableSubstitute { source } => source.to_diagnostic(),
        }
    }
}

/// A source which reads from a `diagnostics::SourceFile`
pub struct FileMapSource {
    src: Arc<SourceFile>,
    bytes: *const [u8],
    start: SourceIndex,
    peek: Option<(SourceIndex, char)>,
    end: usize,
    pos: usize,
    eof: bool,
}
impl FileMapSource {
    fn peek_char(&self) -> Option<(SourceIndex, char)> {
        self.peek
    }

    fn next_char(&mut self) -> Option<(SourceIndex, char)> {
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
                result => result,
            }
        };

        // Reset peek
        self.peek = unsafe { self.next_char_internal() };

        result
    }

    #[inline]
    unsafe fn next_char_internal(&mut self) -> Option<(SourceIndex, char)> {
        let mut pos = self.pos;
        let end = self.end;
        if pos == end {
            self.eof = true;
        }

        if self.eof {
            return None;
        }

        let start = self.start + pos;

        let bytes: &[u8] = &*self.bytes;

        // Decode UTF-8
        let x = *bytes.get_unchecked(pos);
        if x < 128 {
            self.pos = pos + 1;
            return Some((start, char::from_u32_unchecked(x as u32)));
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
    fn utf8_first_byte(byte: u8, width: u32) -> u32 {
        (byte & (0x7F >> width)) as u32
    }

    /// Returns the value of `ch` updated with continuation byte `byte`.
    #[inline]
    fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 {
        (ch << 6) | (byte & Self::CONT_MASK) as u32
    }

    /// Mask of the value bits of a continuation byte.
    const CONT_MASK: u8 = 0b0011_1111;
}
impl Source for FileMapSource {
    fn new(src: Arc<SourceFile>) -> Self {
        let start = SourceIndex::new(src.id(), ByteIndex(0));
        let mut source = Self {
            src,
            bytes: &[],
            peek: None,
            start,
            end: 0,
            pos: 0,
            eof: false,
        };
        let s = source.src.source();
        let bytes = s.as_bytes();
        source.end = bytes.len();
        source.bytes = bytes;
        source.peek = unsafe { source.next_char_internal() };
        source
    }

    #[inline]
    fn read(&mut self) -> Option<(SourceIndex, char)> {
        self.next_char()
    }

    #[inline]
    fn peek(&mut self) -> Option<(SourceIndex, char)> {
        self.peek_char()
    }

    #[inline]
    fn span(&self) -> SourceSpan {
        self.src.source_span()
    }

    #[inline]
    fn slice(&self, span: impl Into<Range<usize>>) -> &str {
        self.src.source_slice(span).unwrap()
    }
}

impl Iterator for FileMapSource {
    type Item = (SourceIndex, char);

    fn next(&mut self) -> Option<Self::Item> {
        self.read()
    }
}

#[cfg(test)]
mod test {
    use pretty_assertions::assert_eq;

    use super::*;

    fn read_all_chars(source: FileMapSource) -> Vec<char> {
        source.map(|result| result.1).collect()
    }

    #[test]
    fn file_source() {
        let expected = vec!['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'];

        let codemap = CodeMap::default();

        let id1 = codemap.add("nofile", "hello world!".to_string());
        let file1 = codemap.get(id1).unwrap();
        let source1 = FileMapSource::new(file1);
        let chars = read_all_chars(source1);

        assert_eq!(expected, chars);

        let id2 = codemap.add("nofile", "hello world!".to_string());
        let file2 = codemap.get(id2).unwrap();
        let mut source2 = FileMapSource::new(file2);
        assert_eq!(
            Some((SourceIndex::new(id2, ByteIndex(0)), 'h')),
            source2.peek()
        );
        assert_eq!(
            Some((SourceIndex::new(id2, ByteIndex(0)), 'h')),
            source2.next()
        );

        let id3 = codemap.add("nofile", "éé".to_string());
        let file3 = codemap.get(id3).unwrap();
        let mut source3 = FileMapSource::new(file3);
        assert_eq!(
            Some((SourceIndex::new(id3, ByteIndex(0)), 'é')),
            source3.peek()
        );
        assert_eq!(
            Some((SourceIndex::new(id3, ByteIndex(0)), 'é')),
            source3.next()
        );
        assert_eq!(
            Some((SourceIndex::new(id3, ByteIndex(2)), 'é')),
            source3.peek()
        );
        assert_eq!(
            Some((SourceIndex::new(id3, ByteIndex(2)), 'é')),
            source3.next()
        );
    }
}
