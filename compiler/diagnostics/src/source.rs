use std::convert::Into;
use std::num::NonZeroU32;
use std::ops::Range;

use super::*;

/// A handle that points to a file in the database.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub(crate) NonZeroU32);
impl SourceId {
    pub(crate) const UNKNOWN_SOURCE_ID: u32 = u32::max_value();

    pub const UNKNOWN: Self = Self(unsafe { NonZeroU32::new_unchecked(Self::UNKNOWN_SOURCE_ID) });

    pub(crate) fn new(index: u32) -> Self {
        assert!(index > 0);
        assert!(index < Self::UNKNOWN_SOURCE_ID);
        Self(NonZeroU32::new(index).unwrap())
    }

    #[inline]
    pub(crate) fn get(self) -> u32 {
        self.0.get()
    }
}

/// The representation of a source file in the database.
#[derive(Debug, Clone)]
pub struct SourceFile {
    id: SourceId,
    name: FileName,
    source: String,
    line_starts: Vec<ByteIndex>,
    parent: Option<SourceSpan>,
}
impl SourceFile {
    pub(crate) fn new(
        id: SourceId,
        name: FileName,
        source: String,
        parent: Option<SourceSpan>,
    ) -> Self {
        let line_starts = codespan_reporting::files::line_starts(source.as_str())
            .map(|i| ByteIndex::from(i as u32))
            .collect();

        Self {
            id,
            name,
            source,
            line_starts,
            parent,
        }
    }

    pub fn name(&self) -> &FileName {
        &self.name
    }

    pub fn id(&self) -> SourceId {
        self.id
    }

    pub fn parent(&self) -> Option<SourceSpan> {
        self.parent
    }

    pub fn line_start(&self, line_index: LineIndex) -> Result<ByteIndex, Error> {
        use std::cmp::Ordering;

        match line_index.cmp(&self.last_line_index()) {
            Ordering::Less => Ok(self.line_starts[line_index.to_usize()]),
            Ordering::Equal => Ok(self.source_span().end_index()),
            Ordering::Greater => Err(Error::LineTooLarge {
                given: line_index.to_usize(),
                max: self.last_line_index().to_usize(),
            }),
        }
    }

    pub fn last_line_index(&self) -> LineIndex {
        LineIndex::from(self.line_starts.len() as RawIndex)
    }

    pub fn line_span(&self, line_index: LineIndex) -> Result<codespan::Span, Error> {
        let line_start = self.line_start(line_index)?;
        let next_line_start = self.line_start(line_index + LineOffset::from(1))?;

        Ok(codespan::Span::new(line_start, next_line_start))
    }

    pub fn line_index(&self, byte_index: ByteIndex) -> LineIndex {
        match self.line_starts.binary_search(&byte_index) {
            // Found the start of a line
            Ok(line) => LineIndex::from(line as u32),
            Err(next_line) => LineIndex::from(next_line as u32 - 1),
        }
    }

    /// Returns a codespan::Span that points to the location given by the provided line:column
    pub fn line_column_to_span(
        &self,
        line_index: LineIndex,
        column_index: ColumnIndex,
    ) -> Result<codespan::Span, Error> {
        let column_index = column_index.to_usize();
        let line_span = self.line_span(line_index)?;
        let line_src = self
            .source
            .as_str()
            .get(line_span.start().to_usize()..line_span.end().to_usize())
            .unwrap();
        if line_src.len() < column_index {
            let base = line_span.start().to_usize();
            return Err(Error::IndexTooLarge {
                given: base + column_index,
                max: base + line_src.len(),
            });
        }
        let (pre, _) = line_src.split_at(column_index);
        let start = line_span.start();
        let offset = ByteOffset::from_str_len(pre);
        Ok(codespan::Span::new(start + offset, start + offset))
    }

    pub fn location<I: Into<ByteIndex>>(&self, byte_index: I) -> Result<Location, Error> {
        let byte_index = byte_index.into();
        let line_index = self.line_index(byte_index);
        let line_start_index = self
            .line_start(line_index)
            .map_err(|_| Error::IndexTooLarge {
                given: byte_index.to_usize(),
                max: self.source().len() - 1,
            })?;
        let line_src = self
            .source
            .as_str()
            .get(line_start_index.to_usize()..byte_index.to_usize())
            .ok_or_else(|| {
                let given = byte_index.to_usize();
                let max = self.source().len() - 1;
                if given >= max {
                    Error::IndexTooLarge { given, max }
                } else {
                    Error::InvalidCharBoundary { given }
                }
            })?;

        Ok(Location {
            line: line_index,
            column: ColumnIndex::from(line_src.chars().count() as u32),
        })
    }

    #[inline(always)]
    pub fn source(&self) -> &str {
        self.source.as_str()
    }

    pub fn source_span(&self) -> SourceSpan {
        SourceSpan {
            source_id: self.id,
            start: ByteIndex(0),
            end: ByteIndex(self.source.len() as u32),
        }
    }

    pub fn source_slice(&self, span: impl Into<Range<usize>>) -> Result<&str, Error> {
        let span = span.into();
        let start = span.start;
        let end = span.end;

        self.source().get(start..end).ok_or_else(|| {
            let max = self.source().len() - 1;
            Error::IndexTooLarge {
                given: if start > max { start } else { end },
                max,
            }
        })
    }
}
