use std::ops::Range;

use firefly_util::diagnostics::*;

use super::source::Source;

/// An implementation of `Scanner` for general use
///
/// source -> pending -> current
///
/// ## pending
/// `peek` returns pending without advancing
///
/// ## current
/// `pop` returns current and advances
/// `read` returns current without advances
pub struct Scanner<S> {
    source: S,
    current: (SourceIndex, char),
    pending: (SourceIndex, char),
    start: SourceIndex,
    end: SourceIndex,
}
impl<S> Scanner<S>
where
    S: Source,
{
    pub fn new(mut source: S) -> Self {
        let span = source.span();
        let start = span.start();
        let end = span.end();
        let current = source.read().unwrap_or((SourceIndex::UNKNOWN, '\0'));
        let pending = source.read().unwrap_or((SourceIndex::UNKNOWN, '\0'));
        Scanner {
            source,
            current,
            pending,
            start,
            end,
        }
    }

    pub fn start(&self) -> SourceIndex {
        self.start
    }

    /// Advance scanner pipeline by a single character.
    ///
    /// Current becomes pending, pending becomes next character from source.
    #[inline]
    pub fn advance(&mut self) {
        self.current = self.pending;
        self.pending = match self.source.read() {
            None => (self.end, '\0'),
            Some(ic) => ic,
        };
    }

    /// Get current character and advance.
    #[inline]
    pub fn pop(&mut self) -> (SourceIndex, char) {
        let current = self.current;
        self.advance();
        current
    }

    /// Get pending character.
    #[inline]
    pub fn peek(&self) -> (SourceIndex, char) {
        self.pending
    }

    /// Get the next character from the source.
    #[inline]
    pub fn peek_next(&mut self) -> (SourceIndex, char) {
        match self.source.peek() {
            None => (self.end, '\0'),
            Some((pos, c)) => (pos, c),
        }
    }

    /// Get current character.
    #[inline]
    pub fn read(&self) -> (SourceIndex, char) {
        self.current
    }

    #[inline]
    pub fn slice(&self, span: impl Into<Range<usize>>) -> &str {
        self.source.slice(span)
    }
}
